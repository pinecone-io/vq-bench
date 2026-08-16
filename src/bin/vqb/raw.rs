//! Compact binary capture of a run's raw outputs (scores, reconstructions, perf),
//! so `vqb eval` can recompute metrics without re-encoding. Little-endian, v1.

use std::io::Write;
use std::path::Path;

use anyhow::{bail, Context, Result};

use crate::results::Timing;

const MAGIC: &[u8; 4] = b"VQBR";
const VERSION: u32 = 4;

pub struct RawData {
    pub meta: RawMeta,
    pub datasets: Vec<RawDataset>,
}

pub struct RawMeta {
    pub name: String,
    pub seed: u64,
    pub ks: Vec<usize>,
    pub temperatures: Vec<f64>,
    pub n_reconstruct: usize,
    pub timestamp: u64,
    /// Worker threads used for encoding.
    pub threads: usize,
    /// Logical CPU cores on the machine that produced this capture.
    pub cores: usize,
    /// Target architecture (`std::env::consts::ARCH`).
    pub arch: String,
    /// Operating system (`std::env::consts::OS`).
    pub os: String,
}

/// The shared, method-independent facts for one dataset: the candidates, the
/// exact scores over them, and the reconstruction references.
pub struct RawDataset {
    pub dataset: String,
    pub dim: usize,
    pub n_base: usize,
    pub n_eval: usize,
    pub n_candidates: usize,
    pub candidates: Vec<Vec<u32>>,
    pub true_scores: Vec<Vec<f32>>,
    pub recon_indices: Vec<u32>,
    pub references: Vec<Vec<f32>>,
    pub methods: Vec<RawMethod>,
}

pub struct RawMethod {
    pub label: String,
    pub bits_per_dim: f64,
    pub model_bits_per_dim: f64,
    pub code_bits_per_dim: f64,
    pub fit_s: f64,
    pub fit_peak_bytes: u64,
    pub encode_s: f64,
    pub encode_peak_bytes: u64,
    pub score_us: Timing,
    pub recon_us: Option<f64>,
    pub approx_scores: Vec<Vec<f32>>,
    pub recons: Option<Vec<Vec<f32>>>,
}

// --- serialize -------------------------------------------------------------

#[derive(Default)]
struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn opt_f64(&mut self, v: Option<f64>) {
        match v {
            Some(x) => {
                self.bool(true);
                self.f64(x);
            }
            None => self.bool(false),
        }
    }
    fn str(&mut self, s: &str) {
        self.u32(s.len() as u32);
        self.0.extend_from_slice(s.as_bytes());
    }
    fn u64_vec(&mut self, v: &[u64]) {
        self.u32(v.len() as u32);
        v.iter().for_each(|&x| self.u64(x));
    }
    fn f64_vec(&mut self, v: &[f64]) {
        self.u32(v.len() as u32);
        v.iter().for_each(|&x| self.f64(x));
    }
    fn u32_vec(&mut self, v: &[u32]) {
        self.u32(v.len() as u32);
        v.iter().for_each(|&x| self.u32(x));
    }
    fn f32_vec(&mut self, v: &[f32]) {
        self.u32(v.len() as u32);
        v.iter().for_each(|&x| self.f32(x));
    }
    fn u32_jagged(&mut self, v: &[Vec<u32>]) {
        self.u32(v.len() as u32);
        v.iter().for_each(|inner| self.u32_vec(inner));
    }
    fn f32_jagged(&mut self, v: &[Vec<f32>]) {
        self.u32(v.len() as u32);
        v.iter().for_each(|inner| self.f32_vec(inner));
    }
    fn timing(&mut self, t: &Timing) {
        self.f64(t.avg);
        self.f64(t.p50);
        self.f64(t.p90);
        self.f64(t.p99);
    }
}

// The layout is written by these three record encoders, shared by the one-shot
// `to_bytes` and the streaming `RawWriter` so both emit byte-identical output.

/// Magic, version, meta, and the dataset count.
fn write_header(w: &mut Writer, m: &RawMeta, n_datasets: u32) {
    w.0.extend_from_slice(MAGIC);
    w.u32(VERSION);
    w.str(&m.name);
    w.u64(m.seed);
    w.u64_vec(&m.ks.iter().map(|&k| k as u64).collect::<Vec<_>>());
    w.f64_vec(&m.temperatures);
    w.u64(m.n_reconstruct as u64);
    w.u64(m.timestamp);
    w.u64(m.threads as u64);
    w.u64(m.cores as u64);
    w.str(&m.arch);
    w.str(&m.os);
    w.u32(n_datasets);
}

/// One dataset's shared facts (candidates, true scores, references) and its
/// method count — the methods themselves follow as `write_method` records.
fn write_dataset_head(w: &mut Writer, d: &RawDataset, n_methods: u32) {
    w.str(&d.dataset);
    w.u64(d.dim as u64);
    w.u64(d.n_base as u64);
    w.u64(d.n_eval as u64);
    w.u64(d.n_candidates as u64);
    w.u32_jagged(&d.candidates);
    w.f32_jagged(&d.true_scores);
    w.u32_vec(&d.recon_indices);
    w.f32_jagged(&d.references);
    w.u32(n_methods);
}

/// One method's capture (perf + raw scores/reconstructions).
fn write_method(w: &mut Writer, m: &RawMethod) {
    w.str(&m.label);
    w.f64(m.bits_per_dim);
    w.f64(m.model_bits_per_dim);
    w.f64(m.code_bits_per_dim);
    w.f64(m.fit_s);
    w.u64(m.fit_peak_bytes);
    w.f64(m.encode_s);
    w.u64(m.encode_peak_bytes);
    w.timing(&m.score_us);
    w.opt_f64(m.recon_us);
    w.f32_jagged(&m.approx_scores);
    match &m.recons {
        Some(r) => {
            w.bool(true);
            w.f32_jagged(r);
        }
        None => w.bool(false),
    }
}

/// Streams a capture to a sink one record at a time, so the runner never holds
/// every method's scores and reconstructions in memory at once. The dataset and
/// method counts are known up front, so the bytes match `to_bytes` exactly and
/// `read` needs no change. Call `begin_dataset` then one `write_method` per
/// method, repeated per dataset, then `finish`.
pub struct RawWriter<W: Write> {
    sink: W,
}

impl<W: Write> RawWriter<W> {
    /// Write the header (magic, version, meta, dataset count) and open the stream.
    pub fn new(mut sink: W, meta: &RawMeta, n_datasets: usize) -> Result<Self> {
        let mut w = Writer::default();
        write_header(&mut w, meta, n_datasets as u32);
        sink.write_all(&w.0).context("writing raw header")?;
        Ok(Self { sink })
    }

    /// Write one dataset's shared facts and its method count. `d.methods` is
    /// ignored — the methods follow via `write_method`.
    pub fn begin_dataset(&mut self, d: &RawDataset, n_methods: usize) -> Result<()> {
        let mut w = Writer::default();
        write_dataset_head(&mut w, d, n_methods as u32);
        self.sink.write_all(&w.0).context("writing raw dataset")?;
        Ok(())
    }

    /// Append one method's capture to the current dataset.
    pub fn write_method(&mut self, m: &RawMethod) -> Result<()> {
        let mut w = Writer::default();
        write_method(&mut w, m);
        self.sink.write_all(&w.0).context("writing raw method")?;
        Ok(())
    }

    /// Flush the sink.
    pub fn finish(mut self) -> Result<()> {
        self.sink.flush().context("flushing raw capture")?;
        Ok(())
    }
}

// --- deserialize -----------------------------------------------------------

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).context("length overflow")?;
        if end > self.buf.len() {
            bail!("unexpected end of raw capture");
        }
        let s = &self.buf[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn bool(&mut self) -> Result<bool> {
        Ok(self.bytes(1)?[0] != 0)
    }
    fn opt_f64(&mut self) -> Result<Option<f64>> {
        Ok(if self.bool()? {
            Some(self.f64()?)
        } else {
            None
        })
    }
    fn str(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        String::from_utf8(self.bytes(n)?.to_vec()).context("invalid utf-8")
    }
    fn u32_vec(&mut self) -> Result<Vec<u32>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.u32()).collect()
    }
    fn f32_vec(&mut self) -> Result<Vec<f32>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.f32()).collect()
    }
    fn u32_jagged(&mut self) -> Result<Vec<Vec<u32>>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.u32_vec()).collect()
    }
    fn f32_jagged(&mut self) -> Result<Vec<Vec<f32>>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| self.f32_vec()).collect()
    }
    fn timing(&mut self) -> Result<Timing> {
        Ok(Timing {
            avg: self.f64()?,
            p50: self.f64()?,
            p90: self.f64()?,
            p99: self.f64()?,
        })
    }
}

/// Parse a capture from bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<RawData> {
    let mut r = Reader { buf: bytes, pos: 0 };
    if r.bytes(4)? != MAGIC {
        bail!("not a vqb raw capture (bad magic)");
    }
    let version = r.u32()?;
    if version != VERSION {
        bail!("unsupported raw version {version}");
    }
    let meta = RawMeta {
        name: r.str()?,
        seed: r.u64()?,
        ks: r.u32_or_u64_ks()?,
        temperatures: {
            let n = r.u32()? as usize;
            (0..n).map(|_| r.f64()).collect::<Result<_>>()?
        },
        n_reconstruct: r.u64()? as usize,
        timestamp: r.u64()?,
        threads: r.u64()? as usize,
        cores: r.u64()? as usize,
        arch: r.str()?,
        os: r.str()?,
    };
    let n_datasets = r.u32()? as usize;
    let mut datasets = Vec::with_capacity(n_datasets);
    for _ in 0..n_datasets {
        let dataset = r.str()?;
        let dim = r.u64()? as usize;
        let n_base = r.u64()? as usize;
        let n_eval = r.u64()? as usize;
        let n_candidates = r.u64()? as usize;
        let candidates = r.u32_jagged()?;
        let true_scores = r.f32_jagged()?;
        let recon_indices = r.u32_vec()?;
        let references = r.f32_jagged()?;
        let n_methods = r.u32()? as usize;
        let mut methods = Vec::with_capacity(n_methods);
        for _ in 0..n_methods {
            methods.push(RawMethod {
                label: r.str()?,
                bits_per_dim: r.f64()?,
                model_bits_per_dim: r.f64()?,
                code_bits_per_dim: r.f64()?,
                fit_s: r.f64()?,
                fit_peak_bytes: r.u64()?,
                encode_s: r.f64()?,
                encode_peak_bytes: r.u64()?,
                score_us: r.timing()?,
                recon_us: r.opt_f64()?,
                approx_scores: r.f32_jagged()?,
                recons: if r.bool()? {
                    Some(r.f32_jagged()?)
                } else {
                    None
                },
            });
        }
        datasets.push(RawDataset {
            dataset,
            dim,
            n_base,
            n_eval,
            n_candidates,
            candidates,
            true_scores,
            recon_indices,
            references,
            methods,
        });
    }
    Ok(RawData { meta, datasets })
}

impl Reader<'_> {
    /// `ks` are written as u64 (count-prefixed).
    fn u32_or_u64_ks(&mut self) -> Result<Vec<usize>> {
        let n = self.u32()? as usize;
        (0..n).map(|_| Ok(self.u64()? as usize)).collect()
    }
}

/// Read a capture from disk.
pub fn read(path: &Path) -> Result<RawData> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    from_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips() {
        let data = RawData {
            meta: RawMeta {
                name: "exp".into(),
                seed: 7,
                ks: vec![1, 10],
                temperatures: vec![0.5, 1.0],
                n_reconstruct: 2,
                timestamp: 123,
                threads: 4,
                cores: 8,
                arch: "x86_64".into(),
                os: "linux".into(),
            },
            datasets: vec![RawDataset {
                dataset: "ds".into(),
                dim: 3,
                n_base: 4,
                n_eval: 2,
                n_candidates: 2,
                candidates: vec![vec![0, 1], vec![2, 3]],
                true_scores: vec![vec![1.0, 0.5], vec![0.2, 0.9]],
                recon_indices: vec![0, 2],
                references: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                methods: vec![RawMethod {
                    label: "minmax[bits=4]".into(),
                    bits_per_dim: 4.0,
                    model_bits_per_dim: 0.01,
                    code_bits_per_dim: 3.99,
                    fit_s: 0.25,
                    fit_peak_bytes: 2048,
                    encode_s: 0.5,
                    encode_peak_bytes: 4096,
                    score_us: Timing {
                        avg: 1.0,
                        p50: 1.0,
                        p90: 2.0,
                        p99: 3.0,
                    },
                    recon_us: Some(0.7),
                    approx_scores: vec![vec![0.9, 0.4], vec![0.1, 0.8]],
                    recons: Some(vec![vec![0.9, 0.1, 0.0], vec![0.0, 0.9, 0.0]]),
                }],
            }],
        };
        // Drive the streaming writer into a buffer, exactly as `run` does to a file.
        let mut buf = Vec::new();
        {
            let mut w = RawWriter::new(&mut buf, &data.meta, data.datasets.len()).unwrap();
            for d in &data.datasets {
                w.begin_dataset(d, d.methods.len()).unwrap();
                for m in &d.methods {
                    w.write_method(m).unwrap();
                }
            }
            w.finish().unwrap();
        }
        let back = from_bytes(&buf).unwrap();
        assert_eq!(back.meta.name, "exp");
        assert_eq!(back.meta.ks, vec![1, 10]);
        assert_eq!(back.meta.threads, 4);
        assert_eq!(back.meta.cores, 8);
        assert_eq!(back.meta.arch, "x86_64");
        assert_eq!(back.meta.os, "linux");
        assert_eq!(back.datasets.len(), 1);
        let d = &back.datasets[0];
        assert_eq!(d.candidates, vec![vec![0, 1], vec![2, 3]]);
        assert_eq!(d.true_scores, vec![vec![1.0, 0.5], vec![0.2, 0.9]]);
        let m = &d.methods[0];
        assert_eq!(m.label, "minmax[bits=4]");
        assert_eq!(m.fit_s, 0.25);
        assert_eq!(m.fit_peak_bytes, 2048);
        assert_eq!(m.encode_peak_bytes, 4096);
        assert_eq!(m.recon_us, Some(0.7));
        assert_eq!(m.approx_scores[1], vec![0.1, 0.8]);
        assert_eq!(m.recons.as_ref().unwrap()[0], vec![0.9, 0.1, 0.0]);
    }
}
