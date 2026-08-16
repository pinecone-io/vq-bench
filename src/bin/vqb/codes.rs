//! On-disk per-method code store, so encoding a huge base can be done once and
//! the codes streamed to disk rather than held in memory as one `Vec<Vec<u8>>`.
//!
//! One self-describing file per (dataset, method), little-endian:
//! ```text
//! magic "VQCS" 4 | version 4 | seed 8 | n_base 8 | dim 8 | n_fit 8 | n_calib 8   (identity, 48 B)
//! variable 8 | width 8 | code_bytes 8 | encode_s 8 | encode_peak_bytes 8         (patched by finish, @48)
//! threads 8 | fit_s 8 | fit_peak_bytes 8                                         (known at create, @88)
//! label_len u32 | label bytes | model_len u32 | model bytes
//! --- data (code_bytes of codes, row order) ---
//! --- lengths (n_base u32s, only when `variable`) ---
//! ```
//! Every in-tree quantizer's code width depends only on the model/dim, so while
//! each code matches the last the file stays fixed-width: row `i` lives at
//! `data_offset + i*width` and no side table is written. The first row that
//! differs switches the file to the variable layout — the rows so far were all
//! `width` bytes, so their lengths follow from the count — and `finish` appends
//! one `u32` length per row. A reader prefix-sums that table into offsets, which
//! cost 8 B/row while the store is open.
//!
//! The header records the full identity of what determines the codes — dataset
//! and method label, `seed`, `n_base`, `n_fit`, `n_calib` — so a reused file can
//! be verified against the current config (see `matches`). `seed`+`n_base` fix
//! the encoded vector set and order (mismatch → wrong metrics); `n_fit`/`n_calib`
//! /`label` fix the model (mismatch → stale but self-consistent results).

use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use anyhow::{bail, ensure, Context, Result};

const MAGIC: &[u8; 4] = b"VQCS";
/// 6, not 5: the variable layout and the fit stats were written independently, each
/// bumping 4 → 5 while claiming different bytes at the same offsets. Stores stamped
/// with either 5 exist, and under a merged 5 they would pass this check and then be
/// read field-for-field wrong. A store is a cache, so orphaning them costs nothing.
const VERSION: u32 = 6;

/// Size of the fixed-length header span (identity, the patched layout/encode stats,
/// `threads`, and the fit stats), i.e. everything before the length-prefixed `label`
/// and `model`. identity 48 (magic 4 + version 4 + seed 8 + n_base 8 + dim 8 +
/// n_fit 8 + n_calib 8) + variable 8 + width 8 + code_bytes 8 + encode_s 8 +
/// encode_peak 8 + threads 8 + fit_s 8 + fit_peak 8.
const FIXED_SPAN: usize = 112;
/// Byte offset of the `variable`/`width`/`code_bytes`/`encode_s`/`encode_peak_bytes`
/// block, patched by `finish` once the layout and encode stats are known.
const PATCH_OFFSET: u64 = 48;
/// Length of that block.
const PATCH_LEN: usize = 40;

/// `<dir>/<dataset>/<method>.codes`. Keyed only by what determines the codes
/// (dataset + method); the rest of the identity is verified via the header.
pub fn path_for(dir: &Path, dataset: &str, label: &str) -> PathBuf {
    dir.join(dataset).join(format!("{}.codes", slug(label)))
}

/// Map a method label like `MinMax (b=2)` to a filesystem-safe stem.
fn slug(label: &str) -> String {
    label
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// What determines a code file's contents, minus the method label: the header
/// fields a stored file must match before `run` may reuse it or `encode` may skip
/// rewriting it. Derivable from a dataset's shapes, so the check can precede the load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Identity {
    pub seed: u64,
    pub n_base: usize,
    pub dim: usize,
    pub n_fit: usize,
    pub n_calib: usize,
}

impl Identity {
    /// The stored codes for `dataset` × `label` under `dir` matching this identity, if
    /// any. A missing, corrupt, or truncated file is a clean miss (`open` validates).
    pub fn stored(&self, dir: &Path, dataset: &str, label: &str) -> Option<CodeStore> {
        CodeStore::open(&path_for(dir, dataset, label))
            .ok()
            .filter(|s| s.matches(self, label))
    }
}

// --- writer ----------------------------------------------------------------

/// Streams per-vector codes to disk, one chunk at a time, patching the header
/// with the observed layout and encode stats at the end. Writes to a sibling
/// `<final>.tmp` and renames into place in `finish`, so an interrupted encode
/// never leaves a partial file at the destination `run` reads.
pub struct CodeWriter {
    w: BufWriter<File>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    n_base: usize,
    /// The width every row has matched so far, or `None` before the first row.
    width: Option<usize>,
    /// Per-row lengths, kept only once a row breaks the uniform width — so a
    /// fixed-width encode carries no per-row memory at all.
    lens: Option<Vec<u32>>,
    count: usize,
    total: u64,
}

impl CodeWriter {
    /// Create the file and write the header (layout/encode stats left as zero
    /// placeholders, filled in by `finish`). `threads` is the encode worker count
    /// in effect and `fit_s`/`fit_peak_bytes` the cost of the `fit` that produced
    /// `model` — metadata only, not part of the code-determining identity.
    pub fn create(
        path: &Path,
        id: &Identity,
        threads: usize,
        fit_s: f64,
        fit_peak_bytes: u64,
        label: &str,
        model: &[u8],
    ) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create {}", parent.display()))?;
        }
        let mut header = Vec::with_capacity(FIXED_SPAN + label.len() + model.len());
        header.extend_from_slice(MAGIC);
        header.extend_from_slice(&VERSION.to_le_bytes());
        header.extend_from_slice(&id.seed.to_le_bytes());
        header.extend_from_slice(&(id.n_base as u64).to_le_bytes());
        header.extend_from_slice(&(id.dim as u64).to_le_bytes());
        header.extend_from_slice(&(id.n_fit as u64).to_le_bytes());
        header.extend_from_slice(&(id.n_calib as u64).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes()); // variable (patched)
        header.extend_from_slice(&0u64.to_le_bytes()); // width (patched)
        header.extend_from_slice(&0u64.to_le_bytes()); // code_bytes (patched)
        header.extend_from_slice(&0f64.to_le_bytes()); // encode_s (patched)
        header.extend_from_slice(&0u64.to_le_bytes()); // encode_peak_bytes (patched)
        header.extend_from_slice(&(threads as u64).to_le_bytes()); // threads (known now)
        header.extend_from_slice(&fit_s.to_le_bytes()); // fit_s (known now)
        header.extend_from_slice(&fit_peak_bytes.to_le_bytes()); // fit_peak_bytes (known now)
        header.extend_from_slice(&(label.len() as u32).to_le_bytes());
        header.extend_from_slice(label.as_bytes());
        header.extend_from_slice(&(model.len() as u32).to_le_bytes());
        header.extend_from_slice(model);

        let tmp_path = tmp_path(path);
        let file =
            File::create(&tmp_path).with_context(|| format!("create {}", tmp_path.display()))?;
        let mut w = BufWriter::new(file);
        w.write_all(&header).context("write codes header")?;
        Ok(Self {
            w,
            tmp_path,
            final_path: path.to_path_buf(),
            n_base: id.n_base,
            width: None,
            lens: None,
            count: 0,
            total: 0,
        })
    }

    /// Append a chunk of per-vector codes, tracking the width for as long as every
    /// row matches and switching to per-row lengths once one doesn't.
    pub fn push_chunk(&mut self, codes: &[Vec<u8>]) -> Result<()> {
        for code in codes {
            ensure!(
                code.len() <= u32::MAX as usize,
                "code row {} is {} bytes, too long to record a length for",
                self.count,
                code.len()
            );
            if let Some(lens) = &mut self.lens {
                lens.push(code.len() as u32);
            } else {
                match self.width {
                    None => self.width = Some(code.len()),
                    // Uniformity just broke. Every row so far was `w` bytes, so the
                    // lengths we never recorded follow from the count alone.
                    Some(w) if w != code.len() => {
                        let mut lens = vec![w as u32; self.count];
                        lens.push(code.len() as u32);
                        self.lens = Some(lens);
                    }
                    Some(_) => {}
                }
            }
            self.w.write_all(code).context("write code")?;
            self.count += 1;
            self.total += code.len() as u64;
        }
        Ok(())
    }

    /// Flush, append the lengths table if the codes were ragged, patch the header
    /// with the layout and encode stats, publish atomically (`fsync` then rename the
    /// `.tmp` into place), and return `(width, code_bytes)` — a `None` width meaning
    /// the rows are addressed by the lengths table rather than by stride.
    pub fn finish(
        mut self,
        encode_s: f64,
        encode_peak_bytes: u64,
    ) -> Result<(Option<usize>, usize)> {
        if self.count != self.n_base {
            bail!(
                "encoded {} rows, expected {} (n_base)",
                self.count,
                self.n_base
            );
        }
        let width = match self.lens.take() {
            None => Some(self.width.unwrap_or(0)),
            Some(lens) => {
                for len in &lens {
                    self.w
                        .write_all(&len.to_le_bytes())
                        .context("write code length")?;
                }
                None
            }
        };
        let mut patch = Vec::with_capacity(PATCH_LEN);
        patch.extend_from_slice(&u64::from(width.is_none()).to_le_bytes());
        patch.extend_from_slice(&(width.unwrap_or(0) as u64).to_le_bytes());
        patch.extend_from_slice(&self.total.to_le_bytes());
        patch.extend_from_slice(&encode_s.to_le_bytes());
        patch.extend_from_slice(&encode_peak_bytes.to_le_bytes());
        // Patch the header in place without unwrapping the `BufWriter`, so `self`
        // stays whole for `Drop` (which reclaims the `.tmp` if we bail below).
        self.w.flush().context("flush codes")?;
        let file = self.w.get_ref();
        file.write_all_at(&patch, PATCH_OFFSET)
            .context("patch codes header")?;
        file.sync_all().context("sync codes")?;
        // The file is complete; publish it atomically. Until this rename the
        // destination is untouched, so an interrupted encode leaves only `.tmp`.
        std::fs::rename(&self.tmp_path, &self.final_path)
            .with_context(|| format!("publish {}", self.final_path.display()))?;
        Ok((width, self.total as usize))
    }
}

impl Drop for CodeWriter {
    /// Best-effort clean up the temp file if `finish` didn't publish it (interrupt,
    /// bail, or early error). A hard kill can still orphan it — fine for a cache.
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.tmp_path);
    }
}

/// The sibling temp path a `CodeWriter` streams to before renaming into place.
fn tmp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(".tmp");
    path.with_file_name(name)
}

// --- reader ----------------------------------------------------------------

/// How a stored row's code is found: by stride when every code came out the same
/// width, else by the stored lengths, prefix-summed into `n + 1` offsets.
enum Layout {
    Fixed(usize),
    Variable(Vec<u64>),
}

/// A read-only handle over an on-disk code file, addressing codes by row index
/// via positioned reads (no full-file load).
pub struct CodeStore {
    file: File,
    seed: u64,
    n: usize,
    dim: usize,
    n_fit: usize,
    n_calib: usize,
    layout: Layout,
    code_bytes: usize,
    data_offset: u64,
    label: String,
    model: Vec<u8>,
    encode_s: f64,
    encode_peak_bytes: u64,
    threads: usize,
    fit_s: f64,
    fit_peak_bytes: u64,
}

impl CodeStore {
    /// Open and validate a code file, reading its header (but not the codes).
    pub fn open(path: &Path) -> Result<Self> {
        let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
        let mut h = [0u8; FIXED_SPAN];
        file.read_exact(&mut h)
            .with_context(|| format!("read header of {}", path.display()))?;
        if &h[0..4] != MAGIC {
            bail!("not a vqb code file (bad magic): {}", path.display());
        }
        let version = u32::from_le_bytes(h[4..8].try_into().unwrap());
        if version != VERSION {
            bail!("unsupported code file version {version}");
        }
        let seed = u64::from_le_bytes(h[8..16].try_into().unwrap());
        let n = u64::from_le_bytes(h[16..24].try_into().unwrap()) as usize;
        let dim = u64::from_le_bytes(h[24..32].try_into().unwrap()) as usize;
        let n_fit = u64::from_le_bytes(h[32..40].try_into().unwrap()) as usize;
        let n_calib = u64::from_le_bytes(h[40..48].try_into().unwrap()) as usize;
        let variable = u64::from_le_bytes(h[48..56].try_into().unwrap()) != 0;
        let width = u64::from_le_bytes(h[56..64].try_into().unwrap()) as usize;
        let code_bytes = u64::from_le_bytes(h[64..72].try_into().unwrap());
        let encode_s = f64::from_le_bytes(h[72..80].try_into().unwrap());
        let encode_peak_bytes = u64::from_le_bytes(h[80..88].try_into().unwrap());
        let threads = u64::from_le_bytes(h[88..96].try_into().unwrap()) as usize;
        let fit_s = f64::from_le_bytes(h[96..104].try_into().unwrap());
        let fit_peak_bytes = u64::from_le_bytes(h[104..112].try_into().unwrap());

        let label = read_len_prefixed(&mut file, "label")?;
        let label = String::from_utf8(label).context("label is not utf-8")?;
        let model = read_len_prefixed(&mut file, "model")?;
        let data_offset = (FIXED_SPAN + 4 + label.len() + 4 + model.len()) as u64;

        // Validate the file length against what the header implies, so a truncated or
        // corrupt file (e.g. a pre-atomic-rename partial encode) is rejected here — the
        // caller opens with `.ok()`, so rejection turns a bad cache into a clean miss.
        let actual = file
            .metadata()
            .with_context(|| format!("stat {}", path.display()))?
            .len();
        // `n`, `width` and `code_bytes` come from the (possibly corrupt) header, so
        // size the file with checked arithmetic — overflow is itself a corruption
        // signal, and a debug-build panic here would dodge the caller's `.ok()`.
        let table = if variable { (n as u64).checked_mul(4) } else { Some(0) };
        let expected = table
            .and_then(|t| code_bytes.checked_add(t))
            .and_then(|tail| tail.checked_add(data_offset));
        if expected != Some(actual) {
            bail!(
                "code file truncated or corrupt: {}: {actual} bytes, expected {}",
                path.display(),
                expected.map_or_else(|| "overflow".to_string(), |e| e.to_string()),
            );
        }
        // Sizing the file passed, so `n` is bounded by it and the table read below
        // cannot be talked into a huge allocation.
        let layout = if variable {
            Layout::Variable(read_offsets(&file, n, data_offset + code_bytes, code_bytes, path)?)
        } else {
            if (n as u64).checked_mul(width as u64) != Some(code_bytes) {
                bail!(
                    "code file header disagrees with itself: {}: \
                     {n} rows × {width} B ≠ {code_bytes} B",
                    path.display()
                );
            }
            Layout::Fixed(width)
        };
        Ok(Self {
            file,
            seed,
            n,
            dim,
            n_fit,
            n_calib,
            layout,
            code_bytes: code_bytes as usize,
            data_offset,
            label,
            model,
            encode_s,
            encode_peak_bytes,
            threads,
            fit_s,
            fit_peak_bytes,
        })
    }

    /// Whether this file was written for the given code-determining identity.
    pub fn matches(&self, id: &Identity, label: &str) -> bool {
        self.seed == id.seed
            && self.n == id.n_base
            && self.dim == id.dim
            && self.n_fit == id.n_fit
            && self.n_calib == id.n_calib
            && self.label == label
    }

    /// Read row `i`'s code into `buf`.
    pub fn get_into(&self, i: usize, buf: &mut Vec<u8>) -> Result<()> {
        let (at, len) = match &self.layout {
            Layout::Fixed(w) => ((i * w) as u64, *w),
            Layout::Variable(offsets) => (offsets[i], (offsets[i + 1] - offsets[i]) as usize),
        };
        buf.resize(len, 0);
        self.file
            .read_exact_at(buf, self.data_offset + at)
            .with_context(|| format!("read code row {i}"))
    }

    /// Read row `i`'s code.
    pub fn get(&self, i: usize) -> Result<Vec<u8>> {
        let mut buf = Vec::new();
        self.get_into(i, &mut buf)?;
        Ok(buf)
    }

    pub fn len(&self) -> usize {
        self.n
    }
    pub fn model(&self) -> &[u8] {
        &self.model
    }
    pub fn code_bytes(&self) -> usize {
        self.code_bytes
    }
    /// Bytes per stored code, or `None` when the rows are ragged and addressed by the
    /// lengths table rather than by a stride — mirroring what `CodeWriter::finish` returns.
    pub fn width(&self) -> Option<usize> {
        match self.layout {
            Layout::Fixed(w) => Some(w),
            Layout::Variable(_) => None,
        }
    }
    pub fn encode_s(&self) -> f64 {
        self.encode_s
    }
    pub fn encode_peak_bytes(&self) -> u64 {
        self.encode_peak_bytes
    }
    pub fn threads(&self) -> usize {
        self.threads
    }
    pub fn fit_s(&self) -> f64 {
        self.fit_s
    }
    pub fn fit_peak_bytes(&self) -> u64 {
        self.fit_peak_bytes
    }
}

/// Read the `n`-entry lengths table at `at` and prefix-sum it into `n + 1` offsets,
/// checking that it accounts for exactly the `code_bytes` of data it describes.
fn read_offsets(
    file: &File,
    n: usize,
    at: u64,
    code_bytes: u64,
    path: &Path,
) -> Result<Vec<u64>> {
    let mut raw = vec![0u8; n * 4];
    file.read_exact_at(&mut raw, at)
        .with_context(|| format!("read code lengths of {}", path.display()))?;
    let mut offsets = Vec::with_capacity(n + 1);
    let mut end = 0u64;
    offsets.push(end);
    for len in raw.chunks_exact(4) {
        end += u64::from(u32::from_le_bytes(len.try_into().unwrap()));
        offsets.push(end);
    }
    if end != code_bytes {
        bail!(
            "code lengths sum to {end} bytes, expected {code_bytes}: {}",
            path.display()
        );
    }
    Ok(offsets)
}

/// Read a `u32`-length-prefixed byte block from `file` at its current cursor.
fn read_len_prefixed(file: &mut File, what: &str) -> Result<Vec<u8>> {
    let mut len = [0u8; 4];
    file.read_exact(&mut len)
        .with_context(|| format!("read {what} length"))?;
    let mut buf = vec![0u8; u32::from_le_bytes(len) as usize];
    file.read_exact(&mut buf)
        .with_context(|| format!("read {what}"))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("vqb_codes_test_{name}.codes"))
    }

    /// A throwaway identity for the tests that only exercise writer/reader mechanics.
    fn stub_id(dim: usize, n_base: usize) -> Identity {
        Identity {
            seed: 1,
            n_base,
            dim,
            n_fit: n_base,
            n_calib: 0,
        }
    }

    #[test]
    fn round_trips_codes_and_header() {
        let path = tmp("round_trip");
        let codes: Vec<Vec<u8>> = (0..10u8).map(|i| vec![i, i.wrapping_add(1), 42]).collect();
        let model = vec![7u8, 8, 9];

        let id = Identity {
            seed: 123,
            n_base: 10,
            dim: 4,
            n_fit: 8,
            n_calib: 5,
        };
        let mut w = CodeWriter::create(&path, &id, 6, 0.25, 2048, "MinMax (b=2)", &model).unwrap();
        w.push_chunk(&codes[..4]).unwrap();
        w.push_chunk(&codes[4..]).unwrap();
        let (width, code_bytes) = w.finish(1.5, 4096).unwrap();
        assert_eq!(width, Some(3));
        assert_eq!(code_bytes, 30);

        let store = CodeStore::open(&path).unwrap();
        assert!(store.matches(&id, "MinMax (b=2)"));
        // Any component of the identity differing rejects the cache.
        assert!(!store.matches(&Identity { dim: 8, ..id }, "MinMax (b=2)"));
        assert!(!store.matches(&Identity { seed: 999, ..id }, "MinMax (b=2)"));
        assert!(!store.matches(&Identity { n_base: 9, ..id }, "MinMax (b=2)"));
        assert!(!store.matches(&Identity { n_fit: 7, ..id }, "MinMax (b=2)"));
        assert!(!store.matches(&Identity { n_calib: 6, ..id }, "MinMax (b=2)"));
        assert!(!store.matches(&id, "MinMax (b=4)"));
        assert_eq!(store.model(), &[7, 8, 9]);
        assert_eq!(store.encode_s(), 1.5);
        assert_eq!(store.encode_peak_bytes(), 4096);
        assert_eq!(store.threads(), 6);
        assert_eq!(store.fit_s(), 0.25);
        assert_eq!(store.fit_peak_bytes(), 2048);
        assert_eq!(store.code_bytes(), 30);
        // Random access by index round-trips every row.
        for i in [0usize, 3, 9, 1] {
            assert_eq!(store.get(i).unwrap(), codes[i]);
        }
        std::fs::remove_file(&path).ok();
    }

    /// `stored` is what `run` reuses and `encode` skips on, so it must hit only on a
    /// full identity match and treat anything else — absent file, wrong seed — as a miss.
    #[test]
    fn stored_hits_only_on_a_full_identity_match() {
        // Its own directory, so the test never writes into a real code store.
        let dir = std::env::temp_dir().join("vqb_codes_test_stored");
        let _ = std::fs::remove_dir_all(&dir); // clear any leftover from a prior crash
        let ds = "stub";
        let label = "MinMax (b=2)";
        let id = Identity {
            seed: 5,
            n_base: 3,
            dim: 2,
            n_fit: 3,
            n_calib: 0,
        };
        assert!(id.stored(&dir, ds, label).is_none(), "no file yet");

        let path = path_for(&dir, ds, label);
        let mut w = CodeWriter::create(&path, &id, 1, 0.0, 0, label, &[]).unwrap();
        w.push_chunk(&[vec![1, 2], vec![3, 4], vec![5, 6]]).unwrap();
        w.finish(0.0, 0).unwrap();

        assert!(id.stored(&dir, ds, label).is_some());
        assert!(id.stored(&dir, ds, "MinMax (b=4)").is_none(), "label");
        assert!(
            Identity { seed: 6, ..id }.stored(&dir, ds, label).is_none(),
            "seed"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Ragged codes round-trip: every row is addressable and the reported size is the
    /// sum of the code lengths, not a width times a count.
    #[test]
    fn round_trips_variable_width_codes() {
        let path = tmp("variable");
        let codes: Vec<Vec<u8>> = (0..10u8).map(|i| vec![i; i as usize % 4 + 1]).collect();
        let total: usize = codes.iter().map(Vec::len).sum();

        let id = Identity {
            seed: 1,
            n_base: codes.len(),
            dim: 4,
            n_fit: 8,
            n_calib: 5,
        };
        let mut w = CodeWriter::create(&path, &id, 2, 0.0, 0, "Stub", &[7]).unwrap();
        w.push_chunk(&codes[..4]).unwrap();
        w.push_chunk(&codes[4..]).unwrap();
        let (width, code_bytes) = w.finish(2.5, 8192).unwrap();
        assert_eq!(width, None, "ragged codes have no single width");
        assert_eq!(code_bytes, total);

        let store = CodeStore::open(&path).unwrap();
        assert_eq!(store.code_bytes(), total);
        assert_eq!(store.model(), &[7]);
        for i in [0usize, 3, 9, 1, 6] {
            assert_eq!(store.get(i).unwrap(), codes[i]);
        }
        std::fs::remove_file(&path).ok();
    }

    /// The writer records no per-row length until one row breaks the uniform width;
    /// the lengths of the rows before it are recovered from the count.
    #[test]
    fn recovers_lengths_of_rows_written_before_the_width_broke() {
        let path = tmp("late_break");
        let mut codes: Vec<Vec<u8>> = (0..8u8).map(|i| vec![i, i]).collect();
        codes.push(vec![99, 99, 99]);
        codes.push(vec![100]);

        let mut w = CodeWriter::create(&path, &stub_id(2, codes.len()), 1, 0.0, 0, "stub", &[]).unwrap();
        // The break lands mid-file, after two whole chunks of uniform rows.
        w.push_chunk(&codes[..4]).unwrap();
        w.push_chunk(&codes[4..]).unwrap();
        let (width, code_bytes) = w.finish(0.0, 0).unwrap();
        assert_eq!(width, None);
        assert_eq!(code_bytes, 8 * 2 + 3 + 1);

        let store = CodeStore::open(&path).unwrap();
        for (i, code) in codes.iter().enumerate() {
            assert_eq!(&store.get(i).unwrap(), code, "row {i}");
        }
        std::fs::remove_file(&path).ok();
    }

    /// A quantizer that owns no per-vector bits stays on the fixed-stride layout —
    /// zero-width codes are uniform, not ragged, so no lengths table is written.
    #[test]
    fn empty_codes_stay_fixed_width() {
        let path = tmp("empty_codes");
        let mut w = CodeWriter::create(&path, &stub_id(2, 3), 1, 0.0, 0, "stub", &[1, 2]).unwrap();
        w.push_chunk(&[vec![], vec![], vec![]]).unwrap();
        let (width, code_bytes) = w.finish(0.0, 0).unwrap();
        assert_eq!((width, code_bytes), (Some(0), 0));

        let store = CodeStore::open(&path).unwrap();
        assert_eq!(store.code_bytes(), 0);
        assert!(store.get(2).unwrap().is_empty());
        std::fs::remove_file(&path).ok();
    }

    /// The lengths table is part of what `open` sizes the file against, so losing it
    /// is a clean miss rather than an out-of-bounds read at score time.
    #[test]
    fn open_rejects_a_variable_store_missing_its_lengths() {
        let path = tmp("variable_truncated");
        let codes: Vec<Vec<u8>> = vec![vec![1], vec![2, 2], vec![3, 3, 3]];
        let mut w = CodeWriter::create(&path, &stub_id(2, codes.len()), 1, 0.0, 0, "stub", &[]).unwrap();
        w.push_chunk(&codes).unwrap();
        w.finish(0.0, 0).unwrap();
        assert!(CodeStore::open(&path).is_ok());

        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 4).unwrap(); // drop one length entry
        drop(f);
        let err = CodeStore::open(&path).err().unwrap();
        assert!(err.to_string().contains("truncated or corrupt"));
        std::fs::remove_file(&path).ok();
    }

    /// A fixed-layout header carries both a width and a total, so `open` refuses one
    /// that contradicts itself instead of trusting a stride into nowhere.
    #[test]
    fn open_rejects_a_fixed_header_that_contradicts_itself() {
        let path = tmp("bad_total");
        let mut w = CodeWriter::create(&path, &stub_id(3, 2), 1, 0.0, 0, "stub", &[]).unwrap();
        w.push_chunk(&[vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        w.finish(0.0, 0).unwrap();
        // Halve the width, leaving `code_bytes` and the file length as they were.
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all_at(&1u64.to_le_bytes(), 56).unwrap(); // width @56
        drop(f);
        let err = CodeStore::open(&path).err().unwrap();
        assert!(err.to_string().contains("disagrees with itself"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn interrupted_encode_leaves_no_destination_file() {
        let path = tmp("interrupted");
        std::fs::remove_file(&path).ok();
        // Encode partially, then drop the writer without `finish` (as a kill would).
        let mut w = CodeWriter::create(&path, &stub_id(3, 4), 1, 0.0, 0, "stub", &[]).unwrap();
        w.push_chunk(&[vec![1, 2, 3], vec![4, 5, 6]]).unwrap();
        drop(w);
        // The destination `run` reads was never created, and `Drop` reclaimed the
        // sibling `.tmp` — nothing is left behind.
        assert!(!path.exists());
        assert!(!tmp_path(&path).exists());
        assert!(CodeStore::open(&path).is_err());
    }

    #[test]
    fn open_rejects_truncated_data() {
        let path = tmp("truncated");
        let codes: Vec<Vec<u8>> = (0..4u8).map(|i| vec![i, i, i]).collect();
        let mut w =
            CodeWriter::create(&path, &stub_id(3, codes.len()), 1, 0.0, 0, "stub", &[]).unwrap();
        w.push_chunk(&codes).unwrap();
        w.finish(0.0, 0).unwrap();
        // A clean file opens; lopping a byte off the data region makes it fail.
        assert!(CodeStore::open(&path).is_ok());
        let len = std::fs::metadata(&path).unwrap().len();
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 1).unwrap();
        drop(f);
        let err = CodeStore::open(&path).err().unwrap();
        assert!(err.to_string().contains("truncated or corrupt"));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn open_rejects_overflowing_header_without_panic() {
        let path = tmp("overflow");
        let mut w = CodeWriter::create(&path, &stub_id(3, 1), 1, 0.0, 0, "stub", &[]).unwrap();
        w.push_chunk(&[vec![1, 2, 3]]).unwrap();
        w.finish(0.0, 0).unwrap();
        // Poison `n` and claim the variable layout, so sizing the lengths table
        // (`n * 4`) would overflow a u64.
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all_at(&u64::MAX.to_le_bytes(), 16).unwrap(); // n_base @16
        f.write_all_at(&1u64.to_le_bytes(), 48).unwrap(); // variable @48
        drop(f);
        let err = CodeStore::open(&path).err().unwrap();
        assert!(err.to_string().contains("truncated or corrupt"));
        std::fs::remove_file(&path).ok();
    }
}
