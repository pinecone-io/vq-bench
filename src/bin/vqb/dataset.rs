//! Dataset download + reformat (`vqb data get`) and loading. VIBE ships
//! `train`/`test`/`learn`/`neighbors`; we reformat to the harness's own layout
//! `db`/`eval`/`calib`/`eval_candidates`, then `load` reads that directly.
//! `eval_candidates` is VIBE's shipped `neighbors` by default, or the exact
//! brute-force top-L (by dot product) when `--candidates L` is passed.

use std::io::Write;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use ndarray::{s, Array2};

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
    let loaded = Loaded {
        base,
        eval,
        calib,
        eval_candidates,
    };
    validate(&loaded).with_context(|| format!("dataset {}", path.display()))?;
    Ok(loaded)
}

/// Reject a dataset whose arrays are empty or mutually inconsistent, so a bad file
/// fails here with a clear message rather than panicking deep in the runner (e.g. a
/// dim-mismatched dot product, or an out-of-range candidate index).
fn validate(l: &Loaded) -> Result<()> {
    validate_parts(&l.base, &l.eval, l.calib.as_ref(), &l.eval_candidates)
}

/// The `validate` checks over borrowed arrays, so the write path can validate before
/// committing to disk without moving its in-memory arrays into a `Loaded`.
fn validate_parts(
    base: &Array2<f32>,
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
    eval_candidates: &[Vec<usize>],
) -> Result<()> {
    let (n_base, dim) = base.dim();
    ensure!(n_base > 0 && dim > 0, "base is empty ({n_base}×{dim})");
    ensure!(eval.nrows() > 0, "eval is empty");
    ensure!(
        eval.ncols() == dim,
        "eval dim {} != base dim {dim}",
        eval.ncols()
    );
    if let Some(c) = calib {
        ensure!(c.nrows() > 0, "calib is empty");
        ensure!(c.ncols() == dim, "calib dim {} != base dim {dim}", c.ncols());
    }
    ensure!(
        eval_candidates.len() == eval.nrows(),
        "eval_candidates has {} rows, expected one per eval query ({})",
        eval_candidates.len(),
        eval.nrows()
    );
    // Every candidate must index into `base`; the `i64 as usize` cast in
    // `read_neighbors` wraps a negative index to a huge value, which this also catches.
    ensure!(
        eval_candidates.iter().flatten().all(|&i| i < n_base),
        "eval_candidates contains an index outside [0, {n_base})"
    );
    Ok(())
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
/// `candidates` sets the `eval_candidates` width L: `None` keeps VIBE's shipped
/// neighbors, `Some(l)` recomputes the exact top-L by brute force. When the file is
/// already local, only the candidates are rebuilt (in place) if L differs.
pub fn get(entry: &Entry, candidates: Option<usize>) -> Result<()> {
    let dest = entry.local_path();
    if dest.exists() {
        if let Some(l) = candidates {
            // `top_neighbors` clamps the stored width to the base size, so compare against the
            // clamped target — otherwise `--candidates L` with L > n_base never converges.
            let target = l.min(stored_base_rows(&dest)?);
            if current_candidate_width(&dest)? != target {
                println!("recomputing {l} candidate(s) for {} ...", dest.display());
                rewrite_candidates(&dest, l)?;
            }
        }
        println!("have {}", dest.display());
        return Ok(());
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent).context("create data dir")?;
    }
    let src = dest.with_extension("src.hdf5");
    download(&entry.url(), &src)?;
    println!("formatting {} ...", dest.display());
    let res = reformat(&src, &dest, candidates);
    // `reformat` writes `dest` atomically (see `write_dataset`), so a failure never
    // leaves a partial file there. Drop `src` once we're done, but keep it around on
    // failure for diagnosis.
    if res.is_ok() {
        let _ = std::fs::remove_file(&src);
    }
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
/// `base←train`, `eval←test`, `calib←learn` (if present). `eval_candidates` is
/// VIBE's `neighbors` when `candidates` is `None`, or the exact brute-force top-L
/// of `test` against `train` when `Some(l)`.
fn reformat(src: &Path, dest: &Path, candidates: Option<usize>) -> Result<()> {
    let inf = hdf5::File::open(src).with_context(|| format!("opening source {}", src.display()))?;
    let train = read_rows(&inf, "train")?;
    let test = read_rows(&inf, "test")?;
    let learn = if has(&inf, "learn") {
        Some(read_rows(&inf, "learn")?)
    } else {
        None
    };
    let nbrs = match candidates {
        Some(l) => top_neighbors(&test, &train, l),
        None if has(&inf, "neighbors") => read_neighbors(&inf, "neighbors")?,
        None => bail!(
            "source {} has no `neighbors`; pass `--candidates L` to brute-force ground truth",
            src.display()
        ),
    };
    write_dataset(dest, &train, &test, learn.as_ref(), &nbrs)
}

/// Write a complete harness dataset to `dest` atomically: build it in a temp file,
/// then rename over `dest` once fully written. An interrupted or failed write thus
/// never leaves a partial file at `dest` that a later `get` would mistake for valid.
fn write_dataset(
    dest: &Path,
    base: &Array2<f32>,
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
    candidates: &[Vec<usize>],
) -> Result<()> {
    // Validate before touching disk, so a bad dataset fails at `data get` time (while
    // the source is still around for diagnosis) rather than on the next run's `load`,
    // and without leaving even a temp file behind.
    validate_parts(base, eval, calib, candidates)?;
    let tmp = dest.with_extension("tmp.hdf5");
    let _ = std::fs::remove_file(&tmp); // clear any leftover from a prior crash
    let build = || -> Result<()> {
        // Scope the file so it is closed (and flushed to disk) before the rename.
        let out =
            hdf5::File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
        write_rows(&out, "base", base)?;
        write_rows(&out, "eval", eval)?;
        if let Some(c) = calib {
            write_rows(&out, "calib", c)?;
        }
        write_neighbors(&out, "eval_candidates", candidates)?;
        Ok(())
    };
    match build() {
        Ok(()) => {
            // hdf5's close flushes but doesn't fsync; sync before the rename so a crash just
            // after can't leave a truncated file at `dest` (mirrors `codes.rs::finish`).
            std::fs::File::open(&tmp)
                .and_then(|f| f.sync_all())
                .with_context(|| format!("syncing {}", tmp.display()))?;
            std::fs::rename(&tmp, dest).context("installing dataset (rename temp)")
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

/// Target size of one eval×base score tile (~256 MB), bounding peak memory when
/// brute-forcing candidates over a full (possibly ~1M-row) base.
const SCORE_TILE_BYTES: usize = 256 << 20;

/// Exact top-`l` base indices (descending dot product) for each eval query. Chunks
/// over eval rows so the score tile never materializes the whole `n_eval × n_base`
/// matrix; picks the top-`l` per row with a partial sort. Clamped to `base.nrows()`.
fn top_neighbors(eval: &Array2<f32>, base: &Array2<f32>, l: usize) -> Vec<Vec<usize>> {
    let n_db = base.nrows();
    let l = l.min(n_db);
    if n_db == 0 {
        return vec![Vec::new(); eval.nrows()];
    }
    let n_eval = eval.nrows();
    let batch = (SCORE_TILE_BYTES / (n_db * 4)).clamp(1, n_eval.max(1));
    let desc = |s: &[f32], a: usize, b: usize| {
        s[b].partial_cmp(&s[a]).unwrap_or(std::cmp::Ordering::Equal)
    };
    // Only worth a progress bar when the work spans more than one tile (a full base);
    // trivial single-tile inputs (and the unit tests) stay silent.
    let show = n_eval > batch;
    // `base.t()` is not row-major, so hoist the transposed copy out of the loop; otherwise
    // `matmul`'s internal `as_standard_layout` rebuilds a full ~n_db×d copy every tile.
    let base_t = base.t().as_standard_layout().to_owned();
    let mut out = Vec::with_capacity(n_eval);
    let mut idx: Vec<usize> = Vec::with_capacity(n_db); // reused across rows
    for start in (0..n_eval).step_by(batch) {
        let end = (start + batch).min(n_eval);
        let scores = vqb::matmul(eval.slice(s![start..end, ..]), base_t.view());
        for row in scores.rows() {
            let s = row.as_slice().expect("contiguous score row");
            idx.clear();
            idx.extend(0..n_db);
            if l < n_db {
                idx.select_nth_unstable_by(l, |&a, &b| desc(s, a, b));
                idx.truncate(l);
            }
            idx.sort_by(|&a, &b| desc(s, a, b));
            out.push(idx.clone());
        }
        if show {
            draw_progress(end, n_eval);
        }
    }
    if show {
        eprintln!();
    }
    out
}

/// Redraw an in-place `[████░░░░] done/total` bar on stderr (carriage return, no newline).
fn draw_progress(done: usize, total: usize) {
    const WIDTH: usize = 20;
    let filled = (done * WIDTH / total.max(1)).min(WIDTH);
    let bar: String = "█".repeat(filled) + &"░".repeat(WIDTH - filled);
    eprint!("\r  candidates [{bar}] {done}/{total}");
    let _ = std::io::stderr().flush();
}

/// Number of `base` rows stored in `path`.
fn stored_base_rows(path: &Path) -> Result<usize> {
    let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let shape = file.dataset("base").context("dataset `base`")?.shape();
    if shape.len() != 2 {
        bail!("`base` must be 2-D, got shape {shape:?}");
    }
    Ok(shape[0])
}

/// Width (L) of the `eval_candidates` already stored in `path`.
fn current_candidate_width(path: &Path) -> Result<usize> {
    let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let ds = file
        .dataset("eval_candidates")
        .context("dataset `eval_candidates`")?;
    let shape = ds.shape();
    if shape.len() != 2 {
        bail!("`eval_candidates` must be 2-D, got shape {shape:?}");
    }
    Ok(shape[1])
}

/// An array's name paired with its `{rows, cols}` shape, or `None` if absent.
pub type ArrayShape = (&'static str, Option<(usize, usize)>);

/// The `{rows, cols}` shape of each stored array; `None` for an absent optional
/// array (only `calib` may be missing).
pub fn array_shapes(path: &Path) -> Result<Vec<ArrayShape>> {
    let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    ["base", "calib", "eval", "eval_candidates"]
        .into_iter()
        .map(|name| {
            if !has(&file, name) {
                return Ok((name, None));
            }
            let shape = file.dataset(name).context(name)?.shape();
            if shape.len() != 2 {
                bail!("`{name}` must be 2-D, got shape {shape:?}");
            }
            Ok((name, Some((shape[0], shape[1]))))
        })
        .collect()
}

/// The row/column counts a code file's identity is derived from.
pub struct Shapes {
    pub base_rows: usize,
    pub dim: usize,
    /// `None` when the dataset has no `calib`.
    pub calib_rows: Option<usize>,
}

/// The identity-determining shapes, read from the HDF5 metadata without loading any
/// rows — so `encode` can find it has nothing to do before paying a multi-GB read.
pub fn identity_shapes(path: &Path) -> Result<Shapes> {
    let shapes = array_shapes(path)?;
    let of = |want| shapes.iter().find(|(n, _)| *n == want).and_then(|&(_, s)| s);
    let (base_rows, dim) = of("base").context("dataset `base` is missing")?;
    Ok(Shapes {
        base_rows,
        dim,
        calib_rows: of("calib").map(|(rows, _)| rows),
    })
}

/// Recompute `eval_candidates` at width `l` from the stored `base`/`eval`. Reads the
/// existing file read-only and writes a fresh one atomically (see `write_dataset`), so
/// an interrupted recompute leaves the previous dataset intact rather than corrupting
/// it in place — at the cost of rewriting the (large) `base`.
fn rewrite_candidates(path: &Path, l: usize) -> Result<()> {
    let (base, eval, calib) = {
        let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let base = read_rows(&file, "base")?;
        let eval = read_rows(&file, "eval")?;
        let calib = has(&file, "calib")
            .then(|| read_rows(&file, "calib"))
            .transpose()?;
        (base, eval, calib)
    };
    let cands = top_neighbors(&eval, &base, l);
    write_dataset(path, &base, &eval, calib.as_ref(), &cands)
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

        reformat(&src, &dest, None).unwrap();
        let loaded = load(&dest).unwrap();
        assert_eq!(loaded.base.dim(), (4, 2));
        assert_eq!(loaded.eval.dim(), (2, 2));
        assert!(loaded.calib.is_none());
        assert_eq!(loaded.eval_candidates, vec![vec![1, 3, 0], vec![2, 3, 0]]);
    }

    /// `array_shapes` reports each stored array's dims; an absent `calib` is `None`.
    #[test]
    fn array_shapes_reports_stored_dims() {
        let dir = std::env::temp_dir().join("vqb-dataset-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("shapes-src.hdf5");
        let dest = dir.join("shapes-formatted.hdf5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dest);

        let f = hdf5::File::create(&src).unwrap();
        let train = Array2::from_shape_vec((4, 2), vec![0., 0., 1., 0., 0., 1., 1., 1.]).unwrap();
        let test = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.2, 0.8]).unwrap();
        write_rows(&f, "train", &train).unwrap();
        write_rows(&f, "test", &test).unwrap();
        write_neighbors(&f, "neighbors", &[vec![1, 3, 0], vec![2, 3, 0]]).unwrap();
        drop(f);

        reformat(&src, &dest, None).unwrap();
        assert_eq!(
            array_shapes(&dest).unwrap(),
            vec![
                ("base", Some((4, 2))),
                ("calib", None),
                ("eval", Some((2, 2))),
                ("eval_candidates", Some((2, 3))),
            ]
        );
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dest).ok();
    }

    /// A bad source (a neighbor index outside `train`) must fail at reformat time,
    /// not silently produce a `dest` that only `load` would later reject.
    #[test]
    fn reformat_rejects_out_of_range_neighbors() {
        let dir = std::env::temp_dir().join("vqb-dataset-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("bad-src.hdf5");
        let dest = dir.join("bad-formatted.hdf5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dest);

        let f = hdf5::File::create(&src).unwrap();
        let train = Array2::from_shape_vec((4, 2), vec![0., 0., 1., 0., 0., 1., 1., 1.]).unwrap();
        let test = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.2, 0.8]).unwrap();
        write_rows(&f, "train", &train).unwrap();
        write_rows(&f, "test", &test).unwrap();
        write_neighbors(&f, "neighbors", &[vec![1, 3, 0], vec![2, 9, 0]]).unwrap(); // 9 >= 4
        drop(f);

        let err = reformat(&src, &dest, None).unwrap_err();
        assert!(err.to_string().contains("outside"));
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dest).ok();
    }

    /// A well-formed `Loaded`: 4×2 base, 2×2 eval, one in-range candidate list per query.
    fn good() -> Loaded {
        Loaded {
            base: Array2::zeros((4, 2)),
            eval: Array2::zeros((2, 2)),
            calib: None,
            eval_candidates: vec![vec![1, 3], vec![2, 0]],
        }
    }

    #[test]
    fn validate_accepts_a_good_dataset() {
        assert!(validate(&good()).is_ok());
    }

    #[test]
    fn validate_rejects_eval_dim_mismatch() {
        let mut l = good();
        l.eval = Array2::zeros((2, 3)); // 3 != base dim 2
        assert!(validate(&l).unwrap_err().to_string().contains("eval dim"));
    }

    #[test]
    fn validate_rejects_calib_dim_mismatch() {
        let mut l = good();
        l.calib = Some(Array2::zeros((2, 3)));
        assert!(validate(&l).unwrap_err().to_string().contains("calib dim"));
    }

    #[test]
    fn validate_rejects_short_candidates() {
        let mut l = good();
        l.eval_candidates = vec![vec![1, 3]]; // 1 row for 2 eval queries
        assert!(validate(&l)
            .unwrap_err()
            .to_string()
            .contains("eval_candidates"));
    }

    #[test]
    fn validate_rejects_out_of_range_candidate() {
        let mut l = good();
        l.eval_candidates = vec![vec![1, 3], vec![2, 4]]; // 4 >= n_base 4
        assert!(validate(&l).unwrap_err().to_string().contains("outside"));
    }

    /// A 4-row base and 2 queries; brute-force top-L must pick the exact descending
    /// neighbors by dot product, and clamp L to the base size.
    #[test]
    fn top_neighbors_picks_exact_descending() {
        let base = Array2::from_shape_vec((4, 2), vec![0., 0., 1., 0., 0., 1., 1., 1.]).unwrap();
        let eval = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.2, 0.8]).unwrap();
        // q0·rows = [0, .9, .1, 1.0] → top2 {3,1}; q1·rows = [0, .2, .8, 1.0] → top2 {3,2}.
        assert_eq!(top_neighbors(&eval, &base, 2), vec![vec![3, 1], vec![3, 2]]);
        // L past the base size clamps to all 4, still fully sorted descending.
        let all = top_neighbors(&eval, &base, 100);
        assert_eq!(all[0], vec![3, 1, 2, 0]);
        assert_eq!(all[1], vec![3, 2, 1, 0]);
    }

    /// `--candidates` ignores the shipped `neighbors` and bakes the brute-force top-L.
    #[test]
    fn reformat_with_candidates_overrides_neighbors() {
        let dir = std::env::temp_dir().join("vqb-dataset-cand-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.hdf5");
        let dest = dir.join("formatted.hdf5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dest);

        let f = hdf5::File::create(&src).unwrap();
        let train = Array2::from_shape_vec((4, 2), vec![0., 0., 1., 0., 0., 1., 1., 1.]).unwrap();
        let test = Array2::from_shape_vec((2, 2), vec![0.9, 0.1, 0.2, 0.8]).unwrap();
        write_rows(&f, "train", &train).unwrap();
        write_rows(&f, "test", &test).unwrap();
        // Deliberately wrong shipped neighbors — must be overridden by --candidates.
        write_neighbors(&f, "neighbors", &[vec![0, 0, 0], vec![0, 0, 0]]).unwrap();
        drop(f);

        reformat(&src, &dest, Some(2)).unwrap();
        let loaded = load(&dest).unwrap();
        assert_eq!(loaded.eval_candidates, vec![vec![3, 1], vec![3, 2]]);

        // Re-widening an existing file rebuilds it atomically at the new width.
        rewrite_candidates(&dest, 3).unwrap();
        assert_eq!(current_candidate_width(&dest).unwrap(), 3);
        let widened = load(&dest).unwrap();
        assert_eq!(widened.eval_candidates, vec![vec![3, 1, 2], vec![3, 2, 1]]);
        // The atomic writer leaves no temp behind on success.
        assert!(!dest.with_extension("tmp.hdf5").exists());

        // Requesting L past the base size clamps the stored width to n_base, so a
        // subsequent `get` sees a matching (clamped) target and does not re-run.
        rewrite_candidates(&dest, 100).unwrap();
        let n_base = stored_base_rows(&dest).unwrap();
        assert_eq!(current_candidate_width(&dest).unwrap(), n_base);
        assert_eq!(current_candidate_width(&dest).unwrap(), 100usize.min(n_base));
    }
}
