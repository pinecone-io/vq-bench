//! `qjl`: unit-normalize, rotate, scale, then 1-bit signs — an unbiased inner-product
//! estimate (the sqrt(2d/pi) scale corrects the sign estimator's bias).

use anyhow::Result;

use super::catalog::{get_or, QuantizerSpec};
use super::Rotation;
use crate::{CastSign, Normalize, Pipeline, Scale};

pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "qjl",
    family: "QJL",
    params: &["rotation"],
    describe: "Normalize -> Rotate -> Scale -> CastSign",
    build: |p, seed, dim| qjl(get_or(p, "rotation", Rotation::Full)?, seed, dim),
};

/// `Normalize -> rotate(seed) -> Scale(sqrt(2d/pi)) -> CastSign`. `d` is the post-rotation
/// width (padded under Hadamard), so the sign estimate is unbiased for either rotation.
pub fn qjl(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
    let stage = rotation.stage(dim, seed);
    let rotated_dim = stage.out_dim(dim);
    let scale = (2.0 * rotated_dim as f32 / std::f32::consts::PI).sqrt();
    Pipeline::new(
        dim,
        vec![
            Box::new(Normalize),
            stage,
            Box::new(Scale::new(scale, 0.0)),
            Box::new(CastSign),
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

    /// The sqrt(2d/pi) scale makes the estimate unbiased: least-squares slope ~ 1.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(7);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            let codec = AsQuantizer(qjl(rotation, 1, d).unwrap());
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
