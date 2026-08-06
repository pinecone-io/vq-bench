//! `rabitq`: center, unit-normalize, rotate, then a 1-bit angular cast — the
//! unbiased RaBitQ inner-product estimate.

use anyhow::Result;

use super::catalog::get_or;
use super::rotation::Rotation;
use crate::{CastAngular, Center, Normalize, Params, Pipeline, Quantizer};

/// The `rabitq` family. 1 bit is hardcoded, so the only param is `rotation`.
pub struct RaBitQ(pub Pipeline);

impl RaBitQ {
    /// `Center -> Normalize -> rotate(seed) -> CastAngular(1)` over input dim `dim`.
    pub fn pipeline(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                rotation.stage(seed),
                Box::new(CastAngular::new(1)),
            ],
        )
    }
}

impl Quantizer for RaBitQ {
    fn name() -> &'static str {
        "rabitq"
    }

    fn params() -> &'static [&'static str] {
        &["rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastAngular(1)"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", Rotation::Hadamard)?;
        Ok(Self(Self::pipeline(rotation, seed, dim)?))
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

    /// The `rabitq` quantizer, with `rotation` given as configs give it.
    fn rabitq(rotation: &str, seed: u64, dim: usize) -> Result<RaBitQ> {
        RaBitQ::build(&params(&[("rotation", json!(rotation))]), seed, dim)
    }

    /// The 1-bit angular estimate is unbiased: the least-squares slope of the
    /// estimate on the true dot is ~ 1, under both rotations.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(5);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (200, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (20, d));
        for rotation in ["full", "hadamard"] {
            let codec = rabitq(rotation, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.25, "slope {}", se / st);
        }
    }
}
