//! `minmax`: per-vector rescale to `[0, 1]`, then a `b`-bit uniform lattice.

use anyhow::{ensure, Result};

use super::catalog::get;
use crate::coding::CodeLayout;
use crate::MinMax as MinMaxStage;
use crate::{CastUint, Params, Pipeline, Quantizer};

/// The `minmax` family. `get` type-checks `b`; value/range checks live in `build`.
pub struct MinMax(pub Pipeline);

impl MinMax {
    /// Rescale each vector to `[0, 1]` (the MinMax stage, aliased — this family
    /// shadows its name — feeding `CastUint` its expected input range), then a
    /// `bits`-bit uniform lattice.
    pub fn pipeline(bits: u8, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Pipeline::new(
            dim,
            vec![Box::new(MinMaxStage::default()), Box::new(CastUint::new(bits))],
        )
    }
}

impl Quantizer for MinMax {
    fn name() -> &'static str {
        "minmax"
    }

    fn params() -> &'static [&'static str] {
        &["b"]
    }

    fn describe() -> &'static str {
        "MinMax -> CastUint(b)"
    }

    fn build(p: &Params, _seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(get(p, "b")?, dim)?))
    }

    crate::pipeline_quantizer!();
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

    /// The `minmax` quantizer with `b = bits` over input dim `dim`.
    fn minmax(bits: u8, dim: usize) -> Result<MinMax> {
        MinMax::build(&params(&[("b", json!(bits))]), 1, dim)
    }

    /// `b` out of `1..=8` is a build error (surfaced by `RunConfig::validate`).
    #[test]
    fn rejects_out_of_range_bits() {
        assert!(minmax(0, 3).is_err());
        assert!(minmax(9, 3).is_err());
        assert!(minmax(1, 3).is_ok());
        assert!(minmax(8, 3).is_ok());
    }

    /// 8-bit round-trip recovers the input within one lattice step, and `score`
    /// tracks the exact dot product to the same tolerance.
    #[test]
    fn roundtrip_and_score() {
        // No constant rows: a zero-range vector is unrecoverable by `MinMax`.
        let v: Array2<f32> = array![[0., 1., 2.], [4., 2., 0.], [-1., 3., 1.], [2., 0., 3.]];
        let q: Array2<f32> = array![[1., 0., -1.], [0.5, 1., 2.]];
        let codec = minmax(8, 3).unwrap();

        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        let recon = codec.reconstruct(&model, &refs(&codes));
        for (x, y) in recon.iter().zip(v.iter()) {
            assert!((x - y).abs() < 0.05, "recon {x} vs {y}");
        }

        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        for (x, y) in est.iter().zip(exact.iter()) {
            assert!((x - y).abs() < 0.2, "score {x} vs {y}");
        }
    }
}
