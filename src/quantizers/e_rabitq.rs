//! `e_rabitq`: center, unit-normalize, rotate, then a b-bit angular cast — the
//! extended (multi-bit) RaBitQ. RaBitQ is the b=1 case.

use anyhow::{ensure, Result};

use super::catalog::{get, get_or, QuantizerSpec};
use super::Rotation;
use crate::coding::CodeLayout;
use crate::{CastAngular, Center, Normalize, Pipeline};

/// The `e_rabitq` family. `get`/`get_or` type-check the params; the `b` range is
/// checked in the builder.
pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "e_rabitq",
    family: "E-RaBitQ",
    params: &["b", "rotation"],
    describe: "Center -> Normalize -> Rotate -> CastAngular(b)",
    build: |p, seed, dim| e_rabitq(get(p, "b")?, get_or(p, "rotation", Rotation::Hadamard)?, seed, dim),
};

/// `Center -> Normalize -> rotate(seed) -> CastAngular(b)` over input dim `dim`.
pub fn e_rabitq(bits: u8, rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
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
            rotation.stage(dim, seed),
            Box::new(CastAngular::new(bits)),
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
        assert!(e_rabitq(0, Rotation::Full, 1, 8).is_err());
        assert!(e_rabitq(9, Rotation::Full, 1, 8).is_err());
        assert!(e_rabitq(1, Rotation::Full, 1, 8).is_ok());
        assert!(e_rabitq(8, Rotation::Full, 1, 8).is_ok());
    }

    /// Unbiased inner-product estimate (slope ~ 1), and higher b recovers direction
    /// more tightly than 1-bit RaBitQ.
    #[test]
    fn unbiased_and_directional() {
        let mut rng = math::seed(7);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            let codec = AsQuantizer(e_rabitq(4, rotation, 1, d).unwrap());
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
