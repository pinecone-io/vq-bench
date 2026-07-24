//! `turboquant_prod`: TurboQuant's product variant — the bulk in a (b-1)-bit Gaussian
//! codebook, one bit reallocated to a 1-bit QJL of the residual for an unbiased
//! inner-product estimate.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or, QuantizerSpec};
use super::qjl::qjl;
use super::Rotation;
use crate::coding::CodeLayout;
use crate::{CastNormal, NormalScale, Normalize, Pipeline};

/// Independent seed offset for the residual's QJL rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0xD15C0;

pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "turboquant_prod",
    family: "TurboQuant-prod",
    params: &["b", "rotation"],
    describe: "Normalize -> Rotate -> CastNormal(b-1, plain) -> QJL",
    build: |p, seed, dim| turboquant_prod(get(p, "b")?, get_or(p, "rotation", Rotation::Full)?, seed, dim),
};

/// `(b-1)`-bit plain Gaussian codebook, then 1-bit QJL on the residual (its own rotation,
/// on the post-rotation width). `b` in `2..=MAX_BITS`.
pub fn turboquant_prod(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
    ensure!(
        (2..=CodeLayout::MAX_BITS).contains(&bits),
        "b must be in 2..={}, got {bits}",
        CodeLayout::MAX_BITS
    );
    let stage = rotation.stage(dim, seed);
    let mid_dim = stage.out_dim(dim); // width the residual lives in (padded under Hadamard)
    let residual = qjl(rotation, seed ^ RESIDUAL_ROTATION_SEED, mid_dim)?;
    Pipeline::new(
        dim,
        vec![
            Box::new(Normalize),
            stage,
            Box::new(CastNormal::new(bits - 1, NormalScale::Plain)),
            Box::new(residual),
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
        assert!(turboquant_prod(1, Rotation::Full, 1, 8).is_err()); // b-1 would be 0
        assert!(turboquant_prod(9, Rotation::Full, 1, 8).is_err());
        assert!(turboquant_prod(2, Rotation::Full, 1, 8).is_ok());
        assert!(turboquant_prod(8, Rotation::Full, 1, 8).is_ok());
    }

    /// The QJL residual bit makes the inner-product estimate unbiased: slope ~ 1.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(13);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            let codec = AsQuantizer(turboquant_prod(4, rotation, 1, d).unwrap());
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.3, "slope {}", se / st);
        }
    }
}
