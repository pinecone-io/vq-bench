//! `eden_prod`: unit-normalize, rotate, then a b-bit Gaussian codebook with the
//! inner-product-unbiased dequant scale (EDEN, unbiased variant).

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastNormal, NormalScale, Normalize, Params, Pipeline, Quantizer};

/// The `eden_prod` family. `get`/`get_or` type-check the params; the `b` range is
/// checked in `build`.
pub struct EdenProd(pub Pipeline);

impl EdenProd {
    /// `Normalize -> rotate(seed) -> CastNormal(b, Unbiased)` over input dim `dim`.
    pub fn pipeline(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Pipeline::new(
            dim,
            vec![
                Box::new(Normalize),
                rotation.stage(seed),
                Box::new(CastNormal::new(bits, NormalScale::Unbiased)),
            ],
        )
    }
}

impl Quantizer for EdenProd {
    fn name() -> &'static str {
        "eden_prod"
    }

    fn display_name() -> &'static str {
        "EDEN-prod"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Normalize -> Rotate -> CastNormal(b, unbiased)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Ok(Self(Self::pipeline(get(p, "b")?, rotation, seed, dim)?))
    }

    crate::pipeline_quantizer!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math;
    use crate::util::testing::params;
    use ndarray::Array2;
    use serde_json::json;

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// The `eden_prod` quantizer, with `rotation` given as configs give it.
    fn eden_prod(bits: u8, rotation: &str, seed: u64, dim: usize) -> Result<EdenProd> {
        let p = params(&[("b", json!(bits)), ("rotation", json!(rotation))]);
        EdenProd::build(&p, seed, dim)
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(eden_prod(0, "full", 1, 8).is_err());
        assert!(eden_prod(9, "full", 1, 8).is_err());
        assert!(eden_prod(1, "full", 1, 8).is_ok());
        assert!(eden_prod(8, "full", 1, 8).is_ok());
    }

    /// Unbiased inner-product estimate: the least-squares slope of the estimate on
    /// the true dot is ~ 1, under both rotations.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(3);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in ["full", "hadamard"] {
            let codec = eden_prod(4, rotation, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.2, "slope {}", se / st);
        }
    }
}
