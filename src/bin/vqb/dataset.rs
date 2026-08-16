//! Dataset download + reformat (`vqb data get`) and loading. VIBE ships
//! `train`/`test`/`learn`/`neighbors`; we reformat to the harness's own layout
//! `db`/`eval`/`calib`/`eval_candidates`, then `load` reads that directly.
//! `eval_candidates` is VIBE's shipped `neighbors` by default, or the exact
//! brute-force top-L (by dot product) when `--candidates L` is passed.
//!
//! `Mode::Stream` keeps the base array on disk throughout, reading and writing it one
//! row block at a time, so peak memory tracks the block budget rather than the dataset.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::io::Write;
use std::path::Path;
use std::process::Command;

use anyhow::{bail, ensure, Context, Result};
use ndarray::{s, Array2, ArrayView2, Axis};

use crate::h5;
use crate::registry::Dataset as Entry;

/// How the base array moves through `data get`: held in RAM whole, or read and written
/// one row block at a time.
#[derive(Clone, Copy)]
pub enum Mode {
    Resident,
    /// Stream the base in blocks of about `block_mb` MiB of f32 rows.
    Stream { block_mb: usize },
}

/// A loaded dataset, ready for the runner.
pub struct Loaded {
    pub base: Base,
    pub eval: Array2<f32>,
    /// Calibration queries for `fit`; `None` when the dataset has no `calib`.
    pub calib: Option<Array2<f32>>,
    /// Per eval query, the base indices of its candidates (top-L neighbors).
    pub eval_candidates: Vec<Vec<usize>>,
}

/// The base vectors a run reads: resident in memory, or left on disk and read in row
/// blocks. Every other array in a dataset is bounded by the query count, so this is the
/// only one that needs the choice.
pub enum Base {
    Mem(Array2<f32>),
    Disk(RowReader),
}

impl Base {
    pub fn nrows(&self) -> usize {
        match self {
            Base::Mem(m) => m.nrows(),
            Base::Disk(r) => r.nrows(),
        }
    }

    pub fn ncols(&self) -> usize {
        match self {
            Base::Mem(m) => m.ncols(),
            Base::Disk(r) => r.ncols(),
        }
    }

    pub fn dim(&self) -> (usize, usize) {
        (self.nrows(), self.ncols())
    }

    /// The whole set as one view, when it is resident — `None` for a streamed base, whose
    /// callers have to work a block at a time.
    pub fn resident(&self) -> Option<ArrayView2<'_, f32>> {
        match self {
            Base::Mem(m) => Some(m.view()),
            Base::Disk(_) => None,
        }
    }

    /// The rows named by `idx`, in `idx` order. `idx` has to stay small — a fit sample, a
    /// query's candidates, the reconstruction sample — since a streamed base pays one
    /// positioned read per row.
    pub fn gather(&self, idx: &[usize]) -> Result<Array2<f32>> {
        match self {
            Base::Mem(m) => Ok(m.select(Axis(0), idx)),
            Base::Disk(r) => r.gather(idx),
        }
    }

    /// Call `f` on each block of at most `rows` rows, in ascending row order. `scratch`
    /// backs a streamed read and is untouched for a resident base; the caller owns it so
    /// the allocation can sit outside whatever it is measuring.
    pub fn for_blocks(
        &self,
        rows: usize,
        scratch: &mut Vec<f32>,
        mut f: impl FnMut(usize, ArrayView2<f32>) -> Result<()>,
    ) -> Result<()> {
        match self {
            Base::Mem(m) => {
                let step = rows.clamp(1, m.nrows().max(1));
                for start in (0..m.nrows()).step_by(step) {
                    let end = (start + step).min(m.nrows());
                    f(start, m.slice(s![start..end, ..]))?;
                }
                Ok(())
            }
            Base::Disk(r) => r.for_blocks(rows, scratch, f),
        }
    }
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

/// A 2-D f32 array left on disk, read a row block (or a scattered row set) at a time.
/// Owns the open file, so the dataset handle outlives whatever opened it.
pub struct RowReader {
    _file: hdf5::File,
    ds: hdf5::Dataset,
    rows: usize,
    dim: usize,
}

impl RowReader {
    /// Open `name` in an already-open file, so the caller can read its small arrays first.
    fn in_file(file: hdf5::File, name: &str) -> Result<Self> {
        let ds = file
            .dataset(name)
            .with_context(|| format!("dataset `{name}`"))?;
        let shape = ds.shape();
        if shape.len() != 2 {
            bail!("`{name}` must be 2-D, got shape {shape:?}");
        }
        Ok(Self {
            _file: file,
            ds,
            rows: shape[0],
            dim: shape[1],
        })
    }

    pub fn nrows(&self) -> usize {
        self.rows
    }

    pub fn ncols(&self) -> usize {
        self.dim
    }

    /// The rows named by `idx`, in `idx` order — one positioned read each, straight into
    /// the destination row.
    fn gather(&self, idx: &[usize]) -> Result<Array2<f32>> {
        let mut flat = vec![0f32; idx.len() * self.dim];
        // Walk the file forwards while filling the caller's row order, so even a shuffled
        // index list reads ascending.
        let mut order: Vec<usize> = (0..idx.len()).collect();
        order.sort_unstable_by_key(|&k| idx[k]);
        for k in order {
            let at = k * self.dim;
            h5::read_rows(&self.ds, idx[k], self.dim, &mut flat[at..at + self.dim])?;
        }
        Array2::from_shape_vec((idx.len(), self.dim), flat).context("reshape gathered rows")
    }

    /// Call `f` on each block of at most `rows` rows, in ascending row order, reading
    /// through `scratch` so one buffer serves a whole pass.
    fn for_blocks(
        &self,
        rows: usize,
        scratch: &mut Vec<f32>,
        mut f: impl FnMut(usize, ArrayView2<f32>) -> Result<()>,
    ) -> Result<()> {
        let step = rows.clamp(1, self.rows.max(1));
        scratch.resize(step * self.dim, 0.0);
        for start in (0..self.rows).step_by(step) {
            let n = step.min(self.rows - start);
            h5::read_rows(&self.ds, start, self.dim, &mut scratch[..n * self.dim])?;
            let view =
                ArrayView2::from_shape((n, self.dim), &scratch[..n * self.dim]).context("block")?;
            f(start, view)?;
        }
        Ok(())
    }
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
/// `Mode::Stream` leaves `base` on disk; the rest is bounded by the query count and stays
/// resident either way.
pub fn load(path: &Path, mode: Mode) -> Result<Loaded> {
    let file = hdf5::File::open(path)
        .with_context(|| format!("opening {} (run `vqb data get` first)", path.display()))?;
    let eval = read_rows(&file, "eval")?;
    let calib = if has(&file, "calib") {
        Some(read_rows(&file, "calib")?)
    } else {
        None
    };
    let eval_candidates = read_neighbors(&file, "eval_candidates")?;
    // Last, so the streamed arm can take the file over once the small arrays are out.
    let base = match mode {
        Mode::Resident => Base::Mem(read_rows(&file, "base")?),
        Mode::Stream { .. } => Base::Disk(RowReader::in_file(file, "base")?),
    };
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
    validate_parts(l.base.dim(), &l.eval, l.calib.as_ref(), &l.eval_candidates)
}

/// The `validate` checks, taking the base as a shape only: the write path validates
/// before committing to disk without moving its arrays into a `Loaded`, and a streamed
/// write never has the base resident to hand at all.
fn validate_parts(
    base: (usize, usize),
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
    eval_candidates: &[Vec<usize>],
) -> Result<()> {
    validate_shapes(base, eval, calib)?;
    validate_candidates(base.0, eval.nrows(), eval_candidates)
}

/// The checks that don't involve the candidate lists, so a streamed write can make them
/// before spending a pass over the base — including where it brute-forces the candidates
/// and has none to check yet.
fn validate_shapes(
    base: (usize, usize),
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
) -> Result<()> {
    let (n_base, dim) = base;
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
    Ok(())
}

/// One candidate list per eval query, every entry indexing into the base.
fn validate_candidates(n_base: usize, n_eval: usize, eval_candidates: &[Vec<usize>]) -> Result<()> {
    ensure!(
        eval_candidates.len() == n_eval,
        "eval_candidates has {} rows, expected one per eval query ({n_eval})",
        eval_candidates.len()
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

/// Download `entry` from VIBE and reformat into the harness layout at `dest`.
/// `candidates` sets the `eval_candidates` width L: `None` keeps VIBE's shipped
/// neighbors, `Some(l)` recomputes the exact top-L by brute force. When the file is
/// already local, only the candidates are rebuilt (in place) if L differs.
pub fn get(entry: &Entry, dest: &Path, candidates: Option<usize>, mode: Mode) -> Result<()> {
    if dest.exists() {
        if let Some(l) = candidates {
            // `top_neighbors` clamps the stored width to the base size, so compare against the
            // clamped target — otherwise `--candidates L` with L > n_base never converges.
            let target = l.min(stored_base_rows(dest)?);
            if current_candidate_width(dest)? != target {
                println!("recomputing {l} candidate(s) for {} ...", dest.display());
                rewrite_candidates(dest, l, mode)?;
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
    let res = reformat(&src, dest, candidates, mode);
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
fn reformat(src: &Path, dest: &Path, candidates: Option<usize>, mode: Mode) -> Result<()> {
    let inf = hdf5::File::open(src).with_context(|| format!("opening source {}", src.display()))?;
    let test = read_rows(&inf, "test")?;
    let learn = if has(&inf, "learn") {
        Some(read_rows(&inf, "learn")?)
    } else {
        None
    };
    // A brute-forced list is left for where the base is read; only the shipped one has
    // to come out of the source here.
    let from = match candidates {
        Some(l) => Candidates::TopL(l),
        None if has(&inf, "neighbors") => Candidates::Given(read_neighbors(&inf, "neighbors")?),
        None => bail!(
            "source {} has no `neighbors`; pass `--candidates L` to brute-force ground truth",
            src.display()
        ),
    };
    match mode {
        Mode::Resident => {
            let train = read_rows(&inf, "train")?;
            let nbrs = match from {
                Candidates::Given(nbrs) => nbrs,
                Candidates::TopL(l) => top_neighbors(&test, &train, l),
            };
            write_dataset(dest, &train, &test, learn.as_ref(), &nbrs)
        }
        Mode::Stream { block_mb } => {
            let train = RowReader::in_file(inf, "train")?;
            stream_dataset(dest, train, &test, learn.as_ref(), from, block_mb)
        }
    }
}

/// Run `build` against a fresh temp file, then rename it over `dest` once fully written.
/// An interrupted or failed build thus never leaves a partial file at `dest` that a later
/// `get` would mistake for valid.
fn atomically(dest: &Path, build: impl FnOnce(&hdf5::File) -> Result<()>) -> Result<()> {
    let tmp = dest.with_extension("tmp.hdf5");
    let _ = std::fs::remove_file(&tmp); // clear any leftover from a prior crash
    // Bind the file inside the match arm so it is closed (and flushed to disk) before
    // the rename below.
    let built = match hdf5::File::create(&tmp) {
        Ok(out) => build(&out),
        Err(e) => Err(e).with_context(|| format!("creating {}", tmp.display())),
    };
    match built {
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

/// Write a complete harness dataset to `dest` atomically, from resident arrays.
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
    validate_parts(base.dim(), eval, calib, candidates)?;
    atomically(dest, |out| {
        write_rows(out, "base", base)?;
        write_rows(out, "eval", eval)?;
        if let Some(c) = calib {
            write_rows(out, "calib", c)?;
        }
        write_neighbors(out, "eval_candidates", candidates)
    })
}

/// Where a rebuild's `eval_candidates` come from.
enum Candidates {
    /// Neighbors already in hand (VIBE's shipped list).
    Given(Vec<Vec<usize>>),
    /// The exact top-L, folded out of the base copy pass.
    TopL(usize),
}

/// Write a complete harness dataset to `dest` atomically, copying `base` out of `src` one
/// row block at a time so it is never resident. `Candidates::TopL` costs nothing extra:
/// the neighbors fall out of the same pass. Needs room for a second copy of the dataset
/// on disk, since the temp file is built alongside `dest`.
fn stream_dataset(
    dest: &Path,
    src: RowReader,
    eval: &Array2<f32>,
    calib: Option<&Array2<f32>>,
    candidates: Candidates,
    block_mb: usize,
) -> Result<()> {
    let shape = (src.nrows(), src.ncols());
    // Everything checkable up front is checked before we spend a pass over the base —
    // the shapes always, since a mismatched `eval` would otherwise surface as a panic
    // inside the first block's matmul. A `TopL` list indexes into the base by
    // construction, so only a `Given` one can also be out of range.
    validate_shapes(shape, eval, calib)?;
    let l = match &candidates {
        Candidates::Given(nbrs) => {
            validate_candidates(shape.0, eval.nrows(), nbrs)?;
            0
        }
        Candidates::TopL(l) => (*l).min(shape.0),
    };
    let block = block_rows(shape.1, eval.nrows(), block_mb);
    let mut top = TopL::new(eval.view(), l);
    atomically(dest, move |out| {
        stream_base(&src, out, &mut top, block)?;
        write_rows(out, "eval", eval)?;
        if let Some(c) = calib {
            write_rows(out, "calib", c)?;
        }
        let nbrs = match candidates {
            Candidates::Given(nbrs) => nbrs,
            Candidates::TopL(_) => top.finish(),
        };
        write_neighbors(out, "eval_candidates", &nbrs)
    })
}

/// Copy `src`'s rows into `out`'s `base` dataset `block` rows at a time, folding each
/// block into `top` as it goes past so the base is read exactly once.
fn stream_base(src: &RowReader, out: &hdf5::File, top: &mut TopL, block: usize) -> Result<()> {
    let (rows, dim) = (src.nrows(), src.ncols());
    let ds = out.new_dataset::<f32>().shape([rows, dim]).create("base")?;
    let show = rows > block;
    let mut scratch = Vec::new();
    src.for_blocks(block, &mut scratch, |start, b| {
        let end = start + b.nrows();
        h5::write_rows(&ds, start, dim, b.as_slice().context("contiguous block")?)?;
        top.push_block(start, b);
        if show {
            draw_progress("base", end, rows);
        }
        Ok(())
    })?;
    if show {
        eprintln!();
    }
    Ok(())
}

/// Rows per streamed block: about `block_mb` MiB of f32 rows, capped so one
/// `n_eval × block` candidate score tile also stays near `SCORE_TILE_BYTES`.
fn block_rows(dim: usize, n_eval: usize, block_mb: usize) -> usize {
    let by_row = (block_mb << 20) / (dim.max(1) * 4);
    by_row.min(tile_rows(n_eval)).max(1)
}

/// Target size of one eval×base score tile (~256 MB), bounding peak memory when
/// brute-forcing candidates over a full (possibly ~1M-row) base.
const SCORE_TILE_BYTES: usize = 256 << 20;

/// Base rows per candidate tile: enough that one `n_eval × rows` f32 score tile stays
/// near `SCORE_TILE_BYTES`.
fn tile_rows(n_eval: usize) -> usize {
    (SCORE_TILE_BYTES / (n_eval.max(1) * 4)).max(1)
}

/// One kept candidate, ordered *best first*: descending score, then ascending index. The
/// index tiebreak is what makes a tiled walk reproducible — with equal dots, whichever
/// block happened to arrive first must not decide the order.
struct Candidate {
    score: f32,
    idx: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .score
            .total_cmp(&self.score)
            .then(self.idx.cmp(&other.idx))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for Candidate {}

/// Each eval query's best `l` candidates so far, folded one base row block at a time so
/// the whole `n_eval × n_base` score matrix never materializes. The heaps are max-heaps
/// under `Candidate`'s best-first order, so the root is the next one to evict.
struct TopL {
    l: usize,
    /// The queries transposed, hoisted out of the block loop: every block multiplies
    /// against them, and `matmul` would otherwise re-standardize them each time.
    queries: Array2<f32>,
    kept: Vec<BinaryHeap<Candidate>>,
}

impl TopL {
    fn new(eval: ArrayView2<f32>, l: usize) -> Self {
        Self {
            l,
            queries: eval.t().as_standard_layout().to_owned(),
            kept: (0..eval.nrows())
                .map(|_| BinaryHeap::with_capacity(l))
                .collect(),
        }
    }

    /// Fold base rows `start..start + block.nrows()` into the running tops. The product
    /// is `block · queriesᵀ` rather than the other way round, so the block — the big
    /// operand — is already in the layout `matmul` wants and is never copied.
    fn push_block(&mut self, start: usize, block: ArrayView2<f32>) {
        if self.l == 0 {
            return;
        }
        let scores = vqb::matmul(block, self.queries.view()); // block rows × n_eval
        for (j, row) in scores.rows().into_iter().enumerate() {
            for (kept, &score) in self.kept.iter_mut().zip(row) {
                let c = Candidate {
                    score,
                    idx: start + j,
                };
                if kept.len() < self.l {
                    kept.push(c);
                } else if kept.peek().is_some_and(|worst| c < *worst) {
                    kept.pop();
                    kept.push(c);
                }
            }
        }
    }

    /// Each query's candidate indices, best first.
    fn finish(self) -> Vec<Vec<usize>> {
        self.finish_scored().0
    }

    /// As `finish`, also handing back the scores it already computed.
    fn finish_scored(self) -> CandidatePools {
        let mut idx = Vec::with_capacity(self.kept.len());
        let mut scores = Vec::with_capacity(self.kept.len());
        for kept in self.kept {
            let best_first = kept.into_sorted_vec();
            idx.push(best_first.iter().map(|c| c.idx).collect());
            scores.push(best_first.into_iter().map(|c| c.score).collect());
        }
        (idx, scores)
    }
}

/// Per eval query: its candidate base indices, best first, and their exact scores.
pub type CandidatePools = (Vec<Vec<usize>>, Vec<Vec<f32>>);

/// Exact top-`l` candidates and their true scores for each eval query, folding the DB one
/// row block at a time. The runner needs this when the searched DB is subsampled, since a
/// dataset's shipped neighbors index the full base.
pub fn top_candidates(
    eval: &Array2<f32>,
    db: &Base,
    l: usize,
    scratch: &mut Vec<f32>,
) -> Result<CandidatePools> {
    let n_db = db.nrows();
    let block = tile_rows(eval.nrows()).min(n_db.max(1));
    let show = n_db > block;
    let mut top = TopL::new(eval.view(), l.min(n_db));
    db.for_blocks(block, scratch, |start, rows| {
        top.push_block(start, rows);
        if show {
            draw_progress("candidates", start + rows.nrows(), n_db);
        }
        Ok(())
    })?;
    if show {
        eprintln!();
    }
    Ok(top.finish_scored())
}

/// Exact top-`l` base indices (descending dot product, ties by ascending index) for each
/// eval query, folding one base row block at a time. Clamped to `base.nrows()`.
fn top_neighbors(eval: &Array2<f32>, base: &Array2<f32>, l: usize) -> Vec<Vec<usize>> {
    let n_db = base.nrows();
    let block = tile_rows(eval.nrows()).min(n_db.max(1));
    let mut top = TopL::new(eval.view(), l.min(n_db));
    // Only worth a progress bar when the work spans more than one tile (a full base);
    // trivial single-tile inputs (and the unit tests) stay silent.
    let show = n_db > block;
    for start in (0..n_db).step_by(block) {
        let end = (start + block).min(n_db);
        top.push_block(start, base.slice(s![start..end, ..]));
        if show {
            draw_progress("candidates", end, n_db);
        }
    }
    if show {
        eprintln!();
    }
    top.finish()
}

/// Redraw an in-place `[████░░░░] done/total` bar on stderr (carriage return, no newline).
fn draw_progress(what: &str, done: usize, total: usize) {
    const WIDTH: usize = 20;
    let filled = (done * WIDTH / total.max(1)).min(WIDTH);
    let bar: String = "█".repeat(filled) + &"░".repeat(WIDTH - filled);
    eprint!("\r  {what} [{bar}] {done}/{total}");
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
/// existing file read-only and writes a fresh one atomically (see `atomically`), so
/// an interrupted recompute leaves the previous dataset intact rather than corrupting
/// it in place — at the cost of rewriting the (large) `base`.
fn rewrite_candidates(path: &Path, l: usize, mode: Mode) -> Result<()> {
    let file = hdf5::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let eval = read_rows(&file, "eval")?;
    let calib = has(&file, "calib")
        .then(|| read_rows(&file, "calib"))
        .transpose()?;
    match mode {
        Mode::Resident => {
            let base = read_rows(&file, "base")?;
            drop(file); // close the source before the rename replaces it
            let cands = top_neighbors(&eval, &base, l);
            write_dataset(path, &base, &eval, calib.as_ref(), &cands)
        }
        Mode::Stream { block_mb } => {
            // `stream_dataset` drops the reader before it renames, so the source stays
            // open only for as long as the copy pass needs it.
            let base = RowReader::in_file(file, "base")?;
            stream_dataset(path, base, &eval, calib.as_ref(), Candidates::TopL(l), block_mb)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A resident base's rows, for comparing two loaded datasets.
    fn rows_of(base: &Base) -> Array2<f32> {
        base.resident().expect("test loads are resident").to_owned()
    }

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

        reformat(&src, &dest, None, Mode::Resident).unwrap();
        let loaded = load(&dest, Mode::Resident).unwrap();
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

        reformat(&src, &dest, None, Mode::Resident).unwrap();
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

        let err = reformat(&src, &dest, None, Mode::Resident).unwrap_err();
        assert!(err.to_string().contains("outside"));
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&dest).ok();
    }

    /// A well-formed `Loaded`: 4×2 base, 2×2 eval, one in-range candidate list per query.
    fn good() -> Loaded {
        Loaded {
            base: Base::Mem(Array2::zeros((4, 2))),
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

        reformat(&src, &dest, Some(2), Mode::Resident).unwrap();
        let loaded = load(&dest, Mode::Resident).unwrap();
        assert_eq!(loaded.eval_candidates, vec![vec![3, 1], vec![3, 2]]);

        // Re-widening an existing file rebuilds it atomically at the new width.
        rewrite_candidates(&dest, 3, Mode::Resident).unwrap();
        assert_eq!(current_candidate_width(&dest).unwrap(), 3);
        let widened = load(&dest, Mode::Resident).unwrap();
        assert_eq!(widened.eval_candidates, vec![vec![3, 1, 2], vec![3, 2, 1]]);
        // The atomic writer leaves no temp behind on success.
        assert!(!dest.with_extension("tmp.hdf5").exists());

        // Requesting L past the base size clamps the stored width to n_base, so a
        // subsequent `get` sees a matching (clamped) target and does not re-run.
        rewrite_candidates(&dest, 100, Mode::Resident).unwrap();
        let n_base = stored_base_rows(&dest).unwrap();
        assert_eq!(current_candidate_width(&dest).unwrap(), n_base);
        assert_eq!(current_candidate_width(&dest).unwrap(), 100usize.min(n_base));
    }

    /// Equal dot products must break by ascending index, or a tiled walk's answer would
    /// depend on which block happened to arrive first.
    #[test]
    fn top_neighbors_breaks_ties_by_index() {
        // Rows 0..3 are identical, so every dot with `eval` ties at 1.0.
        let base = Array2::from_shape_vec((4, 2), vec![1., 0., 1., 0., 1., 0., 1., 0.]).unwrap();
        let eval = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        assert_eq!(top_neighbors(&eval, &base, 2), vec![vec![0, 1]]);
        // A one-row tile forces four separate blocks; the answer must not move.
        let mut top = TopL::new(eval.view(), 2);
        for i in 0..4 {
            top.push_block(i, base.slice(s![i..i + 1, ..]));
        }
        assert_eq!(top.finish(), vec![vec![0, 1]]);
    }

    /// A block-at-a-time fold agrees with the whole-base one for every block size.
    #[test]
    fn tiled_and_whole_base_folds_agree() {
        let base = Array2::from_shape_fn((9, 3), |(i, j)| ((i * 7 + j * 5) % 11) as f32 - 5.0);
        let eval = Array2::from_shape_fn((3, 3), |(i, j)| ((i * 3 + j) % 5) as f32 - 2.0);
        let want = top_neighbors(&eval, &base, 4);
        for block in [1usize, 2, 4, 9, 100] {
            let mut top = TopL::new(eval.view(), 4);
            for start in (0..9).step_by(block) {
                let end = (start + block).min(9);
                top.push_block(start, base.slice(s![start..end, ..]));
            }
            assert_eq!(top.finish(), want, "block {block}");
        }
    }

    /// `top_candidates` hands back the scores alongside the indices, descending within
    /// each query, and clamps `l` to the DB size.
    #[test]
    fn top_candidates_returns_scores_with_indices() {
        // One query [1,0]; dots with db rows: 0.0, 1.0, 0.5, -1.0 → top-2 is rows 1 then 2.
        let eval = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();
        let db = Base::Mem(
            Array2::from_shape_vec((4, 2), vec![0., 1., 1., 0., 0.5, 0., -1., 0.]).unwrap(),
        );
        let (cands, truths) = top_candidates(&eval, &db, 2, &mut Vec::new()).unwrap();
        assert_eq!(cands, vec![vec![1, 2]]);
        assert_eq!(truths, vec![vec![1.0, 0.5]]);
        assert!(truths[0].windows(2).all(|w| w[0] >= w[1]));

        let two = Base::Mem(Array2::from_shape_vec((2, 2), vec![1., 0., 0., 1.]).unwrap());
        let (clamped, _) = top_candidates(&eval, &two, 100, &mut Vec::new()).unwrap();
        assert_eq!(clamped[0].len(), 2);
    }

    /// A streamed base must reach the same candidates, gathers and blocks as a resident one.
    #[test]
    fn streamed_base_matches_a_resident_one() {
        let dir = std::env::temp_dir().join("vqb-dataset-base-test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("base.hdf5");
        let _ = std::fs::remove_file(&path);

        let rows = Array2::from_shape_fn((12, 4), |(i, j)| ((i * 5 + j * 3) % 7) as f32 - 3.0);
        let eval = Array2::from_shape_fn((3, 4), |(i, j)| ((i * 3 + j) % 5) as f32 - 2.0);
        write_dataset(&path, &rows, &eval, None, &[vec![0], vec![1], vec![2]]).unwrap();

        let mem = load(&path, Mode::Resident).unwrap().base;
        let disk = load(&path, Mode::Stream { block_mb: 1 }).unwrap().base;
        assert_eq!(disk.dim(), mem.dim());
        assert!(disk.resident().is_none());

        // A shuffled, repeating index list gathers to the same rows either way.
        let idx = [7usize, 0, 11, 3, 7];
        assert_eq!(disk.gather(&idx).unwrap(), mem.gather(&idx).unwrap());

        // And every block size walks the same rows in the same order.
        for block in [1usize, 5, 12, 100] {
            let mut seen = (Vec::new(), Vec::new());
            for (base, into) in [(&mem, &mut seen.0), (&disk, &mut seen.1)] {
                base.for_blocks(block, &mut Vec::new(), |start, b| {
                    into.push((start, b.to_owned()));
                    Ok(())
                })
                .unwrap();
            }
            assert_eq!(seen.0, seen.1, "block {block}");
        }

        // Candidates agree too, streamed or not.
        let want = top_candidates(&eval, &mem, 5, &mut Vec::new()).unwrap();
        assert_eq!(top_candidates(&eval, &disk, 5, &mut Vec::new()).unwrap(), want);
        std::fs::remove_file(&path).ok();
    }

    /// A streamed `data get` must produce exactly the file a resident one does — same
    /// base rows, same brute-forced candidates — for every block size.
    #[test]
    fn streamed_reformat_matches_resident() {
        let dir = std::env::temp_dir().join("vqb-dataset-stream-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.hdf5");
        let train = Array2::from_shape_fn((10, 3), |(i, j)| ((i * 5 + j * 3) % 7) as f32 - 3.0);
        let test = Array2::from_shape_fn((3, 3), |(i, j)| ((i * 4 + j) % 5) as f32 - 2.0);
        let learn = Array2::from_shape_fn((4, 3), |(i, j)| (i + j) as f32);
        let _ = std::fs::remove_file(&src);
        let f = hdf5::File::create(&src).unwrap();
        write_rows(&f, "train", &train).unwrap();
        write_rows(&f, "test", &test).unwrap();
        write_rows(&f, "learn", &learn).unwrap();
        write_neighbors(&f, "neighbors", &[vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        drop(f);

        let resident = dir.join("resident.hdf5");
        let _ = std::fs::remove_file(&resident);
        reformat(&src, &resident, Some(4), Mode::Resident).unwrap();
        let want = load(&resident, Mode::Resident).unwrap();

        // `block_mb` is a byte budget, so 0 MiB exercises the one-row floor and a large
        // value the whole-array case.
        for block_mb in [0usize, 1, 256] {
            let dest = dir.join(format!("streamed-{block_mb}.hdf5"));
            let _ = std::fs::remove_file(&dest);
            reformat(&src, &dest, Some(4), Mode::Stream { block_mb }).unwrap();
            let got = load(&dest, Mode::Resident).unwrap();
            assert_eq!(rows_of(&got.base), rows_of(&want.base), "block_mb {block_mb}");
            assert_eq!(got.eval, want.eval, "block_mb {block_mb}");
            assert_eq!(got.calib, want.calib, "block_mb {block_mb}");
            assert_eq!(got.eval_candidates, want.eval_candidates, "block_mb {block_mb}");
            assert!(!dest.with_extension("tmp.hdf5").exists());
            std::fs::remove_file(&dest).ok();
        }

        // The shipped-neighbors path streams the base too, without touching the list.
        let shipped = dir.join("shipped.hdf5");
        let _ = std::fs::remove_file(&shipped);
        reformat(&src, &shipped, None, Mode::Stream { block_mb: 1 }).unwrap();
        let got = load(&shipped, Mode::Resident).unwrap();
        assert_eq!(rows_of(&got.base), rows_of(&want.base));
        assert_eq!(got.eval_candidates, vec![vec![1, 2], vec![3, 4], vec![5, 6]]);

        // And a streamed re-widening rebuilds in place at the new width.
        rewrite_candidates(&shipped, 6, Mode::Stream { block_mb: 1 }).unwrap();
        let widened = load(&shipped, Mode::Resident).unwrap();
        assert_eq!(rows_of(&widened.base), rows_of(&want.base));
        assert_eq!(widened.eval_candidates, top_neighbors(&test, &train, 6));
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&resident).ok();
        std::fs::remove_file(&shipped).ok();
    }

    /// A streamed write rejects an out-of-range shipped neighbor before it spends a pass
    /// over the base, so nothing lands at `dest`.
    #[test]
    fn streamed_reformat_rejects_out_of_range_neighbors() {
        let dir = std::env::temp_dir().join("vqb-dataset-stream-bad-test");
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
        write_neighbors(&f, "neighbors", &[vec![1, 3, 0], vec![2, 9, 0]]).unwrap(); // 9 >= 4
        drop(f);

        let err = reformat(&src, &dest, None, Mode::Stream { block_mb: 1 }).unwrap_err();
        assert!(err.to_string().contains("outside"));
        assert!(!dest.exists());
        assert!(!dest.with_extension("tmp.hdf5").exists());
        std::fs::remove_file(&src).ok();
    }

    /// Brute-forcing the candidates leaves no list to check, but the shapes still have to
    /// hold: a `test` that doesn't match `train` would otherwise reach the first block's
    /// matmul and panic there, mid-write.
    #[test]
    fn streamed_reformat_rejects_a_dim_mismatch_while_brute_forcing() {
        let dir = std::env::temp_dir().join("vqb-dataset-stream-dim-test");
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.hdf5");
        let dest = dir.join("formatted.hdf5");
        let _ = std::fs::remove_file(&src);
        let _ = std::fs::remove_file(&dest);
        // The temp path too: without the check below, this case panicked mid-write and
        // orphaned one, which is exactly the litter the check is here to keep out.
        let _ = std::fs::remove_file(dest.with_extension("tmp.hdf5"));

        let f = hdf5::File::create(&src).unwrap();
        write_rows(&f, "train", &Array2::zeros((4, 2))).unwrap();
        write_rows(&f, "test", &Array2::zeros((2, 3))).unwrap(); // 3 columns, not 2
        drop(f);

        let err = reformat(&src, &dest, Some(2), Mode::Stream { block_mb: 1 }).unwrap_err();
        assert!(err.to_string().contains("eval dim 3 != base dim 2"), "{err}");
        assert!(!dest.exists());
        assert!(!dest.with_extension("tmp.hdf5").exists());
        std::fs::remove_file(&src).ok();
    }
}
