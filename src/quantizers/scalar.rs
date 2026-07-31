//! `scalar`: per-dimension rescale to `[0, 1]` (calibrated over the fit set), then a
//! `b`-bit uniform lattice.

use anyhow::{ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::get;
use crate::coding::CodeLayout;
use crate::{CastUint, MinMaxDim, Params, Pipeline, Primitive, Quantizer};

/// The `scalar` family. `get` type-checks `b`; value/range checks live in `build`.
pub struct Scalar(pub Pipeline);

impl Scalar {
    /// Rescale each dimension to `[0, 1]` (`MinMaxDim`, feeding `CastUint` its
    /// expected input range), then a `bits`-bit uniform lattice.
    pub fn pipeline(bits: u8, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Pipeline::new(
            dim,
            vec![Box::new(MinMaxDim::default()), Box::new(CastUint::new(bits))],
        )
    }
}

impl Quantizer for Scalar {
    fn name() -> &'static str {
        "scalar"
    }

    fn params() -> &'static [&'static str] {
        &["b"]
    }

    fn describe() -> &'static str {
        "MinMaxDim -> CastUint(b)"
    }

    fn build(p: &Params, _seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(get(p, "b")?, dim)?))
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::params;
    use ndarray::{array, Array2};
    use serde_json::json;

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// The `scalar` quantizer with `b = bits` over input dim `dim`.
    fn scalar(bits: u8, dim: usize) -> Result<Scalar> {
        Scalar::build(&params(&[("b", json!(bits))]), 1, dim)
    }

    /// `b` out of `1..=8` is a build error (surfaced by `RunConfig::validate`).
    #[test]
    fn rejects_out_of_range_bits() {
        assert!(scalar(0, 3).is_err());
        assert!(scalar(9, 3).is_err());
        assert!(scalar(1, 3).is_ok());
        assert!(scalar(8, 3).is_ok());
    }

    /// 8-bit round-trip recovers the input within one per-dimension lattice step, and
    /// `score` tracks the exact dot product to a matching tolerance.
    #[test]
    fn roundtrip_and_score() {
        // No constant columns: a zero-span dimension is unrecoverable by `MinMaxDim`.
        let v: Array2<f32> = array![[0., 1., 2.], [4., 2., 0.], [-1., 3., 1.], [2., 0., 3.]];
        let q: Array2<f32> = array![[1., 0., -1.], [0.5, 1., 2.]];
        let codec = scalar(8, 3).unwrap();

        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        // Per-dim span here is <= 5, so an 8-bit half-bin is well under 0.05.
        let recon = codec.reconstruct(&model, &refs(&codes));
        for (x, y) in recon.iter().zip(v.iter()) {
            assert!((x - y).abs() < 0.05, "recon {x} vs {y}");
        }

        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        for (x, y) in est.iter().zip(exact.iter()) {
            assert!((x - y).abs() < 0.3, "score {x} vs {y}");
        }
    }
}
