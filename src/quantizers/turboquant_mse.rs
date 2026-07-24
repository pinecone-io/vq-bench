//! `turboquant_mse`: unit-normalize, rotate, then a b-bit Gaussian codebook with the
//! plain (S=1) dequant scale (TurboQuant, MSE variant — EDEN's pipeline, plain scale).

use anyhow::{ensure, Result};

use super::catalog::{get, get_or, QuantizerSpec};
use super::Rotation;
use crate::coding::CodeLayout;
use crate::{CastNormal, NormalScale, Normalize, Pipeline};

/// The `turboquant_mse` family. `get`/`get_or` type-check the params; the `b` range
/// is checked in the builder.
pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "turboquant_mse",
    family: "TurboQuant-MSE",
    params: &["b", "rotation"],
    describe: "Normalize -> Rotate -> CastNormal(b, plain)",
    build: |p, seed, dim| turboquant_mse(get(p, "b")?, get_or(p, "rotation", Rotation::Full)?, seed, dim),
};

/// `Normalize -> rotate(seed) -> CastNormal(b, Plain)` over input dim `dim`.
pub fn turboquant_mse(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
    ensure!(
        (1..=CodeLayout::MAX_BITS).contains(&bits),
        "b must be in 1..={}, got {bits}",
        CodeLayout::MAX_BITS
    );
    Pipeline::new(
        dim,
        vec![
            Box::new(Normalize),
            rotation.stage(dim, seed),
            Box::new(CastNormal::new(bits, NormalScale::Plain)),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math, AsQuantizer, Quantizer};
    use ndarray::Array2;

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(turboquant_mse(0, Rotation::Full, 1, 8).is_err());
        assert!(turboquant_mse(9, Rotation::Full, 1, 8).is_err());
        assert!(turboquant_mse(1, Rotation::Full, 1, 8).is_ok());
        assert!(turboquant_mse(8, Rotation::Full, 1, 8).is_ok());
    }

    /// The asymmetric score equals the exact dot with the pipeline's own
    /// reconstruction, under both rotations.
    #[test]
    fn score_tracks_reconstruction() {
        let mut rng = math::seed(2);
        let v: Array2<f32> = math::gaussian(&mut rng, (40, 128));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, 128));
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            let codec = AsQuantizer(turboquant_mse(6, rotation, 1, 128).unwrap());
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let r = refs(&codes);
            let recon = codec.reconstruct(&model, &r);
            let est = codec.score(&model, q.view(), &r);
            let exact = q.dot(&recon.t());
            for (a, b) in est.iter().zip(exact.iter()) {
                assert!((a - b).abs() < 1e-2, "score {a} vs {b}");
            }
        }
    }
}
