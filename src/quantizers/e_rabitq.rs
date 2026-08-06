//! `e_rabitq`: center, unit-normalize, rotate, then a b-bit angular cast — the
//! extended (multi-bit) RaBitQ. RaBitQ is the b=1 case.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or};
use super::rotation::Rotation;
use crate::coding::CodeLayout;
use crate::{CastAngular, Center, Normalize, Params, Pipeline, Quantizer};

/// The `e_rabitq` family. `get`/`get_or` type-check the params; the `b` range is
/// checked in `build`.
pub struct ERaBitQ(pub Pipeline);

impl ERaBitQ {
    /// `Center -> Normalize -> rotate(seed) -> CastAngular(b)` over input dim `dim`.
    pub fn pipeline(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                rotation.stage(seed),
                Box::new(CastAngular::new(bits)),
            ],
        )
    }
}

impl Quantizer for ERaBitQ {
    fn name() -> &'static str {
        "e_rabitq"
    }

    fn display_name() -> &'static str {
        "E-RaBitQ"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> Rotate -> CastAngular(b)"
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

    /// The `e_rabitq` quantizer, with `rotation` given as configs give it.
    fn e_rabitq(bits: u8, rotation: &str, seed: u64, dim: usize) -> Result<ERaBitQ> {
        let p = params(&[("b", json!(bits)), ("rotation", json!(rotation))]);
        ERaBitQ::build(&p, seed, dim)
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(e_rabitq(0, "full", 1, 8).is_err());
        assert!(e_rabitq(9, "full", 1, 8).is_err());
        assert!(e_rabitq(1, "full", 1, 8).is_ok());
        assert!(e_rabitq(8, "full", 1, 8).is_ok());
    }

    /// Unbiased inner-product estimate (slope ~ 1), and higher b recovers direction
    /// more tightly than 1-bit RaBitQ.
    #[test]
    fn unbiased_and_directional() {
        let mut rng = math::seed(7);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in ["full", "hadamard"] {
            let codec = e_rabitq(4, rotation, 1, d).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let est = codec.score(&model, q.view(), &refs(&codes));
            let exact = q.dot(&v.t());
            let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
            let st: f32 = exact.iter().map(|t| t * t).sum();
            assert!(((se / st) - 1.0).abs() < 0.15, "slope {}", se / st);
        }
    }
}
