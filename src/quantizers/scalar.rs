//! `scalar`: per-dimension rescale to `[0, 1]` (calibrated over the fit set), then a
//! `b`-bit uniform lattice. Scalar quantization -- the per-vector `minmax` done per
//! coordinate instead.

use anyhow::{ensure, Result};

use super::catalog::{get, QuantizerSpec};
use crate::coding::CodeLayout;
use crate::{CastUint, MinMaxDim, Pipeline};

/// The `scalar` family. `get` type-checks `b` (inferred as `u8` from `scalar`);
/// value/range checks live in the builder.
pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "scalar",
    family: "Scalar",
    params: &["b"],
    describe: "MinMaxDim -> CastUint(b)",
    build: |p, _seed, dim| scalar(get(p, "b")?, dim),
};

/// The `scalar` pipeline over input dim `dim`: rescale each dimension to `[0, 1]`
/// (`MinMaxDim`, feeding `CastUint` its expected input range), then a `bits`-bit
/// uniform lattice (`CastUint`; `bits` is config key `b`).
pub fn scalar(bits: u8, dim: usize) -> Result<Pipeline> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AsQuantizer, Quantizer};
    use ndarray::{array, Array2};

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
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
        let codec = AsQuantizer(scalar(8, 3).unwrap());

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
