//! On-disk per-method code store, so encoding a huge base can be done once and
//! the codes streamed to disk rather than held in memory as one `Vec<Vec<u8>>`.
//!
//! One self-describing file per (dataset, method), little-endian:
//! ```text
//! magic "VQCS" 4 | version 4 | seed 8 | n_base 8 | dim 8 | n_fit 8 | n_calib 8   (identity, 48 B)
//! width 8 | encode_s 8 | encode_peak_bytes 8                                     (patched by finish, @48)
//! threads 8                                                                      (encode worker count, @72)
//! label_len u32 | label bytes | model_len u32 | model bytes
//! --- data (n_base × width bytes of codes, row order) ---
//! ```
//! Codes are fixed-width (every in-tree quantizer's code width depends only on
//! the model/dim, not the data), so row `i` lives at `data_offset + i*width` and
//! no offset table is needed. The writer captures the width from the first code
//! and errors if any later code differs.
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

use anyhow::{bail, Context, Result};

const MAGIC: &[u8; 4] = b"VQCS";
const VERSION: u32 = 4;

/// Size of the fixed-length header span (identity + the patched width/stats +
/// `threads`), i.e. everything before the length-prefixed `label` and `model`.
/// identity 48 (magic 4 + version 4 + seed 8 + n_base 8 + dim 8 + n_fit 8 + n_calib 8)
/// + width 8 + encode_s 8 + encode_peak 8 + threads 8.
const FIXED_SPAN: usize = 80;
/// Byte offset of the `width`/`encode_s`/`encode_peak_bytes` triple, patched by
/// `finish` once the width and encode stats are known.
const PATCH_OFFSET: u64 = 48;

/// `results/codes/<dataset>/<method>.codes`. Keyed only by what determines the
/// codes (dataset + method); the rest of the identity is verified via the header.
pub fn path_for(dataset: &str, label: &str) -> PathBuf {
    Path::new("results/codes")
        .join(dataset)
        .join(format!("{}.codes", slug(label)))
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

// --- writer ----------------------------------------------------------------

/// Streams per-vector codes to disk, one chunk at a time, patching the header
/// with the observed width and encode stats at the end. Writes to a sibling
/// `<final>.tmp` and renames into place in `finish`, so an interrupted encode
/// never leaves a partial file at the destination `run` reads.
pub struct CodeWriter {
    w: BufWriter<File>,
    tmp_path: PathBuf,
    final_path: PathBuf,
    n_base: usize,
    width: Option<usize>,
    count: usize,
}

impl CodeWriter {
    /// Create the file and write the header (width/encode stats left as zero
    /// placeholders, filled in by `finish`). `n_fit`/`n_calib` are the resolved
    /// row counts used to fit the model; `threads` is the encode worker count in
    /// effect (metadata only — not part of the code-determining identity).
    #[allow(clippy::too_many_arguments)]
    pub fn create(
        path: &Path,
        seed: u64,
        dim: usize,
        n_base: usize,
        n_fit: usize,
        n_calib: usize,
        threads: usize,
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
        header.extend_from_slice(&seed.to_le_bytes());
        header.extend_from_slice(&(n_base as u64).to_le_bytes());
        header.extend_from_slice(&(dim as u64).to_le_bytes());
        header.extend_from_slice(&(n_fit as u64).to_le_bytes());
        header.extend_from_slice(&(n_calib as u64).to_le_bytes());
        header.extend_from_slice(&0u64.to_le_bytes()); // width (patched)
        header.extend_from_slice(&0f64.to_le_bytes()); // encode_s (patched)
        header.extend_from_slice(&0u64.to_le_bytes()); // encode_peak_bytes (patched)
        header.extend_from_slice(&(threads as u64).to_le_bytes()); // threads (known now)
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
            n_base,
            width: None,
            count: 0,
        })
    }

    /// Append a chunk of per-vector codes, enforcing a uniform width.
    pub fn push_chunk(&mut self, codes: &[Vec<u8>]) -> Result<()> {
        for code in codes {
            match self.width {
                Some(w) if code.len() != w => bail!(
                    "codes-on-disk requires a fixed code width: row {} is {} bytes, expected {}",
                    self.count,
                    code.len(),
                    w
                ),
                Some(_) => {}
                None => self.width = Some(code.len()),
            }
            self.w.write_all(code).context("write code")?;
            self.count += 1;
        }
        Ok(())
    }

    /// Flush, patch the header with the width and encode stats, publish atomically
    /// (`fsync` then rename the `.tmp` into place), and return `(width, code_bytes)`.
    pub fn finish(mut self, encode_s: f64, encode_peak_bytes: u64) -> Result<(usize, usize)> {
        if self.count != self.n_base {
            bail!(
                "encoded {} rows, expected {} (n_base)",
                self.count,
                self.n_base
            );
        }
        let width = self.width.unwrap_or(0);
        let mut patch = Vec::with_capacity(24);
        patch.extend_from_slice(&(width as u64).to_le_bytes());
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
        Ok((width, width * self.count))
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

/// A read-only handle over an on-disk code file, addressing codes by row index
/// via positioned reads (no full-file load).
pub struct CodeStore {
    file: File,
    seed: u64,
    n: usize,
    dim: usize,
    n_fit: usize,
    n_calib: usize,
    width: usize,
    data_offset: u64,
    label: String,
    model: Vec<u8>,
    encode_s: f64,
    encode_peak_bytes: u64,
    threads: usize,
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
        let width = u64::from_le_bytes(h[48..56].try_into().unwrap()) as usize;
        let encode_s = f64::from_le_bytes(h[56..64].try_into().unwrap());
        let encode_peak_bytes = u64::from_le_bytes(h[64..72].try_into().unwrap());
        let threads = u64::from_le_bytes(h[72..80].try_into().unwrap()) as usize;

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
        // `n` and `width` come from the (possibly corrupt) header, so size the file
        // with checked arithmetic — overflow is itself a corruption signal, and a
        // debug-build panic here would dodge the caller's `.ok()`.
        let expected = (n as u64)
            .checked_mul(width as u64)
            .and_then(|data| data.checked_add(data_offset));
        if expected != Some(actual) {
            bail!(
                "code file truncated or corrupt: {}: {actual} bytes, expected {}",
                path.display(),
                expected.map_or_else(|| "overflow".to_string(), |e| e.to_string()),
            );
        }
        Ok(Self {
            file,
            seed,
            n,
            dim,
            n_fit,
            n_calib,
            width,
            data_offset,
            label,
            model,
            encode_s,
            encode_peak_bytes,
            threads,
        })
    }

    /// Whether this file was written for the given code-determining identity.
    #[allow(clippy::too_many_arguments)]
    pub fn matches(
        &self,
        seed: u64,
        n_base: usize,
        dim: usize,
        n_fit: usize,
        n_calib: usize,
        label: &str,
    ) -> bool {
        self.seed == seed
            && self.n == n_base
            && self.dim == dim
            && self.n_fit == n_fit
            && self.n_calib == n_calib
            && self.label == label
    }

    /// Read row `i`'s code into `buf`.
    pub fn get_into(&self, i: usize, buf: &mut Vec<u8>) -> Result<()> {
        buf.resize(self.width, 0);
        self.file
            .read_exact_at(buf, self.data_offset + (i * self.width) as u64)
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
        self.n * self.width
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

    #[test]
    fn round_trips_codes_and_header() {
        let path = tmp("round_trip");
        let codes: Vec<Vec<u8>> = (0..10u8).map(|i| vec![i, i.wrapping_add(1), 42]).collect();
        let model = vec![7u8, 8, 9];

        let mut w =
            CodeWriter::create(&path, 123, 4, codes.len(), 8, 5, 6, "MinMax (b=2)", &model).unwrap();
        w.push_chunk(&codes[..4]).unwrap();
        w.push_chunk(&codes[4..]).unwrap();
        let (width, code_bytes) = w.finish(1.5, 4096).unwrap();
        assert_eq!(width, 3);
        assert_eq!(code_bytes, 30);

        let store = CodeStore::open(&path).unwrap();
        assert!(store.matches(123, 10, 4, 8, 5, "MinMax (b=2)"));
        // Any component of the identity differing rejects the cache.
        assert!(!store.matches(123, 10, 8, 8, 5, "MinMax (b=2)")); // dim
        assert!(!store.matches(999, 10, 4, 8, 5, "MinMax (b=2)")); // seed
        assert!(!store.matches(123, 10, 4, 7, 5, "MinMax (b=2)")); // n_fit
        assert!(!store.matches(123, 10, 4, 8, 6, "MinMax (b=2)")); // n_calib
        assert!(!store.matches(123, 10, 4, 8, 5, "MinMax (b=4)")); // label
        assert_eq!(store.model(), &[7, 8, 9]);
        assert_eq!(store.encode_s(), 1.5);
        assert_eq!(store.encode_peak_bytes(), 4096);
        assert_eq!(store.threads(), 6);
        assert_eq!(store.code_bytes(), 30);
        // Random access by index round-trips every row.
        for i in [0usize, 3, 9, 1] {
            assert_eq!(store.get(i).unwrap(), codes[i]);
        }
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_nonuniform_width() {
        let path = tmp("nonuniform");
        let mut w = CodeWriter::create(&path, 1, 2, 2, 2, 0, 1, "stub", &[]).unwrap();
        let err = w.push_chunk(&[vec![1, 2], vec![3]]).unwrap_err();
        assert!(err.to_string().contains("fixed code width"));
    }

    #[test]
    fn interrupted_encode_leaves_no_destination_file() {
        let path = tmp("interrupted");
        std::fs::remove_file(&path).ok();
        // Encode partially, then drop the writer without `finish` (as a kill would).
        let mut w = CodeWriter::create(&path, 1, 3, 4, 4, 0, 1, "stub", &[]).unwrap();
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
        let mut w = CodeWriter::create(&path, 1, 3, codes.len(), 4, 0, 1, "stub", &[]).unwrap();
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
        let mut w = CodeWriter::create(&path, 1, 3, 1, 1, 0, 1, "stub", &[]).unwrap();
        w.push_chunk(&[vec![1, 2, 3]]).unwrap();
        w.finish(0.0, 0).unwrap();
        // Poison the header's `n` and `width` so `n * width` would overflow a u64.
        let f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        f.write_all_at(&u64::MAX.to_le_bytes(), 16).unwrap(); // n_base @16
        f.write_all_at(&u64::MAX.to_le_bytes(), 48).unwrap(); // width @48
        drop(f);
        let err = CodeStore::open(&path).err().unwrap();
        assert!(err.to_string().contains("truncated or corrupt"));
        std::fs::remove_file(&path).ok();
    }
}
