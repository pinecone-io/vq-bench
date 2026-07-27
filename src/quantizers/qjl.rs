//! `qjl`: unit-normalize, rotate into `b*dim` dimensions, then 1-bit signs — an unbiased
//! inner-product estimate (CastSign's sqrt(pi/2m) scale corrects the sign estimator's bias).

use anyhow::{ensure, Result};

use super::catalog::{get_or, QuantizerSpec};
use super::simhash::coded_dim;
use super::Rotation;
use crate::{CastSign, Normalize, Pipeline, Primitive, Resize};

pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "qjl",
    family: "QJL",
    params: &["b", "rotation"],
    describe: "Normalize -> Rotate to b*dim dims -> CastSign",
    build: |p, seed, dim| {
        qjl(
            get_or(p, "b", 1.0f32)?,
            get_or(p, "rotation", Rotation::Hadamard)?,
            seed,
            dim,
        )
    },
};

/// `Normalize -> rotate(seed) to m = b*dim dims -> CastSign`. The projection count is QJL's
/// native bit budget. CastSign scores with the QJL scale sqrt(pi/2m) on the width it is
/// handed, so the sign estimate is unbiased for either rotation and any `b`.
pub fn qjl(bits: f32, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
    ensure!(bits.is_finite() && bits > 0.0, "b must be positive, got {bits}");
    let m = coded_dim(bits, dim);
    let wide = dim.max(m);
    let stage = rotation.stage(wide, seed);
    let rotated = stage.out_dim(wide); // padded to a multiple of 64 under Hadamard
    let mut stages: Vec<Box<dyn Primitive>> = vec![Box::new(Normalize)];
    if wide != dim {
        stages.push(Box::new(Resize::new(dim, wide))); // pad up, so the added dims rotate in
    }
    stages.push(stage);
    if m != dim && rotated != m {
        stages.push(Box::new(Resize::new(rotated, m))); // truncate down to the budget
    }
    stages.push(Box::new(CastSign));
    Pipeline::new(dim, stages)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math, AsQuantizer, Quantizer};
    use ndarray::Array2;

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// The sqrt(2d/pi) scale makes the estimate unbiased: least-squares slope ~ 1. Holds at
    /// any budget, since CastSign reads the width it is handed.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(7);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            for bits in [0.5f32, 1.0, 2.0] {
                let codec = AsQuantizer(qjl(bits, rotation, 1, d).unwrap());
                let model = codec.fit(v.view(), None);
                let codes = codec.encode(&model, v.view());
                let est = codec.score(&model, q.view(), &refs(&codes));
                let exact = q.dot(&v.t());
                let se: f32 = est.iter().zip(exact.iter()).map(|(e, t)| e * t).sum();
                let st: f32 = exact.iter().map(|t| t * t).sum();
                assert!(((se / st) - 1.0).abs() < 0.3, "b={bits} slope {}", se / st);
            }
        }
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(qjl(0.0, Rotation::Full, 1, 8).is_err());
        assert!(qjl(f32::NAN, Rotation::Full, 1, 8).is_err());
        assert!(qjl(0.5, Rotation::Full, 1, 8).is_ok());
    }
}
