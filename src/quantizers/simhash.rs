//! `simhash`: center, unit-normalize, rotate into `b*dim` dimensions, then 1-bit signs
//! scored by a Hamming-angle inner-product estimate.

use anyhow::{bail, ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::get_or;
use crate::{
    CastHamming, Center, Normalize, Params, Pipeline, Primitive, Quantizer, RandomHadamard,
    RandomRotate, Resize,
};

/// The `simhash` family. One sign bit per coded dimension, so `b` bits per input
/// dimension means `m = b*dim` random hyperplanes; `b` may be fractional.
pub struct SimHash(pub Pipeline);

impl SimHash {
    /// `Center -> Normalize -> rotate(seed) to m = b*dim dims -> CastHamming` over
    /// input dim `dim`.
    pub fn pipeline(bits: f32, rotation: &str, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(bits.is_finite() && bits > 0.0, "b must be positive, got {bits}");
        let m = coded_dim(bits, dim);
        let wide = dim.max(m);
        let stage: Box<dyn Primitive> = match rotation {
            "full" => Box::new(RandomRotate::new(seed)),
            "hadamard" => Box::new(RandomHadamard::new(wide, seed)),
            other => bail!("unknown rotation `{other}` (expected `full` or `hadamard`)"),
        };
        let rotated = stage.out_dim(wide); // padded to a multiple of 64 under Hadamard
        let mut stages: Vec<Box<dyn Primitive>> = vec![Box::new(Center), Box::new(Normalize)];
        if wide != dim {
            stages.push(Box::new(Resize::new(dim, wide))); // pad up, so the added dims rotate in
        }
        stages.push(stage);
        if m != dim && rotated != m {
            stages.push(Box::new(Resize::new(rotated, m))); // truncate down to the budget
        }
        stages.push(Box::new(CastHamming));
        Pipeline::new(dim, stages)
    }
}

impl Quantizer for SimHash {
    fn name() -> &'static str {
        "simhash"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate to b*dim dims -> CastHamming"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", "hadamard".to_string())?;
        Ok(Self(Self::pipeline(get_or(p, "b", 1.0f32)?, &rotation, seed, dim)?))
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        self.0.fit(vectors, queries)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        self.0.encode(model, vectors)
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        self.0.reconstruct(model, codes, None)
    }

    fn score(&self, model: &[u8], queries: ArrayView2<f32>, codes: &[&[u8]]) -> Array2<f32> {
        self.0.score(model, queries, codes, None)
    }
}

/// Coded dims for a budget of `bits` sign bits per input dim: `m == dim` at `b == 1`,
/// which leaves the rotation's own width (pad included) untouched.
pub(super) fn coded_dim(bits: f32, dim: usize) -> usize {
    ((bits * dim as f32).round() as usize).max(1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::params;
    use crate::{byte_split, math};
    use ndarray::{s, Array2};
    use serde_json::json;

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// The `simhash` quantizer, with `rotation` given as configs give it.
    fn simhash(bits: f32, rotation: &str, seed: u64, dim: usize) -> Result<SimHash> {
        let p = params(&[("b", json!(bits)), ("rotation", json!(rotation))]);
        SimHash::build(&p, seed, dim)
    }

    /// Top-1 hits over 8 queries, each a noised copy of one base vector.
    fn hits(bits: f32, rotation: &str, d: usize) -> usize {
        let mut rng = math::seed(42);
        let v: Array2<f32> = math::gaussian(&mut rng, (40, d));
        let q = &v.slice(s![0..8, ..]).to_owned() + &(0.3 * math::gaussian(&mut rng, (8, d)));
        let codec = simhash(bits, rotation, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        (0..8)
            .filter(|&i| {
                let row = est.row(i);
                (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap() == i
            })
            .count()
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(simhash(0.0, "full", 1, 8).is_err());
        assert!(simhash(-1.0, "full", 1, 8).is_err());
        assert!(simhash(f32::NAN, "full", 1, 8).is_err());
        assert!(simhash(0.25, "full", 1, 8).is_ok());
        assert!(simhash(16.0, "full", 1, 8).is_ok());
    }

    /// Each query is a noised copy of one base vector and must score it highest.
    #[test]
    fn recovers_nearest_neighbor() {
        for rotation in ["full", "hadamard"] {
            assert!(hits(1.0, rotation, 128) >= 7, "top-1 hits");
        }
    }

    /// b == 1 is the original pipeline: no resize stage, so the rotation keeps its own
    /// width (padded under Hadamard) and encode does no extra passes.
    #[test]
    fn b1_adds_no_resize_stage() {
        assert_eq!(coded_dim(1.0, 128), 128);
        let d = 96; // Hadamard pads to 128, which b == 1 must leave alone
        let v: Array2<f32> = math::gaussian(&mut math::seed(5), (4, d));
        for (rotation, want) in [("full", d), ("hadamard", 128)] {
            let codec = simhash(1.0, rotation, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let per_vector = byte_split(&model, &codes).1 / 4;
            assert_eq!(per_vector, want.div_ceil(8) + 4, "b=1 kept width");
        }
    }

    /// The code lands on exactly m bits per vector (plus Normalize's 4-byte norm) for
    /// either rotation, including the fractional and wider-than-dim cases.
    #[test]
    fn code_bytes_track_the_bit_budget() {
        let d = 96; // not a multiple of 64: Hadamard pads to 128 internally
        let v: Array2<f32> = math::gaussian(&mut math::seed(5), (4, d));
        for rotation in ["full", "hadamard"] {
            for bits in [0.25f32, 0.5, 2.0, 4.0] {
                let m = (bits * d as f32).round() as usize;
                let codec = simhash(bits, rotation, 1, d).unwrap();
                let model = codec.fit(v.view(), None);
                let codes = codec.encode(&model, v.view());
                let per_vector = byte_split(&model, &codes).1 / 4;
                assert_eq!(per_vector, m.div_ceil(8) + 4, "b={bits} m={m}");
            }
        }
    }

    /// Relative squared error of the inner-product estimate over a random batch.
    fn score_error(bits: f32, rotation: &str, d: usize) -> f32 {
        let mut rng = math::seed(11);
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        let codec = simhash(bits, rotation, 1, d).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        let err: f32 = est.iter().zip(exact.iter()).map(|(e, t)| (e - t) * (e - t)).sum();
        err / exact.iter().map(|t| t * t).sum::<f32>()
    }

    /// More hyperplanes estimate the inner product more tightly; fewer degrade it. This is
    /// the property the bit budget buys, and it holds for either rotation.
    #[test]
    fn error_falls_with_the_bit_budget() {
        for rotation in ["full", "hadamard"] {
            let errs: Vec<f32> = [0.25f32, 1.0, 4.0]
                .iter()
                .map(|&b| score_error(b, rotation, 256))
                .collect();
            assert!(errs[1] < errs[0], "b=1 err {} not below b=0.25 {}", errs[1], errs[0]);
            assert!(errs[2] < errs[1], "b=4 err {} not below b=1 {}", errs[2], errs[1]);
        }
    }
}
