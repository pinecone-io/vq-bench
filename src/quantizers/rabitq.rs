//! `rabitq`: center, unit-normalize, rotate, then a 1-bit angular cast — the
//! unbiased RaBitQ inner-product estimate.

use anyhow::Result;

use super::catalog::{get_or, QuantizerSpec};
use super::Rotation;
use crate::{CastAngular, Center, Normalize, Pipeline};

/// The `rabitq` family. 1 bit is hardcoded, so the only param is `rotation`.
pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "rabitq",
    family: "RaBitQ",
    params: &["rotation"],
    describe: "Center -> Normalize -> Rotate -> CastAngular(1)",
    build: |p, seed, dim| rabitq(get_or(p, "rotation", Rotation::Full)?, seed, dim),
};

/// `Center -> Normalize -> rotate(seed) -> CastAngular(1)` over input dim `dim`.
pub fn rabitq(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
    Pipeline::new(
        dim,
        vec![
            Box::new(Center),
            Box::new(Normalize),
            rotation.stage(dim, seed),
            Box::new(CastAngular::new(1)),
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

    /// The 1-bit angular estimate is unbiased: the least-squares slope of the
    /// estimate on the true dot is ~ 1, under both rotations.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(5);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (200, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (20, d));
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            let codec = AsQuantizer(rabitq(rotation, 1, d).unwrap());
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
