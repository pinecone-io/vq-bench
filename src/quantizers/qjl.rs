//! `qjl`: unit-normalize, rotate into `b*dim` dimensions, then 1-bit signs — an unbiased
//! inner-product estimate (CastSign's sqrt(pi/2m) scale corrects the sign estimator's bias).

use anyhow::{bail, ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::get_or;
use super::simhash::coded_dim;
use crate::{
    CastSign, Normalize, Params, Pipeline, Primitive, Quantizer, RandomHadamard, RandomRotate,
    Resize,
};

/// The `qjl` family. The projection count `m = b*dim` is QJL's native bit budget;
/// CastSign scores with the QJL scale sqrt(pi/2m) on the width it is handed, so the
/// sign estimate is unbiased for either rotation and any `b`.
pub struct Qjl(pub Pipeline);

impl Qjl {
    /// `Normalize -> rotate(seed) to m = b*dim dims -> CastSign` over input dim `dim`.
    pub fn pipeline(bits: f32, rotation: &str, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(bits.is_finite() && bits > 0.0, "b must be positive, got {bits}");
        let m = coded_dim(bits, dim);
        let wide = dim.max(m);
        let stage: Box<dyn Primitive> = match rotation {
            "full" => Box::new(RandomRotate::new(seed)),
            "hadamard" => Box::new(RandomHadamard::new(wide, seed)),
            other => bail!("unknown rotation `{other}` (expected `full` or `hadamard`)"),
        };
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
}

impl Quantizer for Qjl {
    fn name() -> &'static str {
        "qjl"
    }

    fn display_name() -> &'static str {
        "QJL"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Normalize -> Rotate to b*dim dims -> CastSign"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", "hadamard".to_string())?;
        Ok(Self(Self::pipeline(get_or(p, "b", 1.0f32)?, &rotation, seed, dim)?))
    }

    fn fit(&self, vectors: ArrayView2<f32>, queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        self.0.fit(vectors, queries)
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        self.0.encode(model, vectors)
    }

    fn reconstruct(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        self.0.reconstruct(model, codes, None)
    }

    fn score(&self, model: &[u8], queries: ArrayView2<f32>, codes: &[&[u8]]) -> Array2<f32> {
        self.0.score(model, queries, codes, None)
    }
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

    /// The `qjl` quantizer, with `rotation` given as configs give it.
    fn qjl(bits: f32, rotation: &str, seed: u64, dim: usize) -> Result<Qjl> {
        let p = params(&[("b", json!(bits)), ("rotation", json!(rotation))]);
        Qjl::build(&p, seed, dim)
    }

    /// The sqrt(2d/pi) scale makes the estimate unbiased: least-squares slope ~ 1. Holds at
    /// any budget, since CastSign reads the width it is handed.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(7);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in ["full", "hadamard"] {
            for bits in [0.5f32, 1.0, 2.0] {
                let codec = qjl(bits, rotation, 1, d).unwrap();
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
        assert!(qjl(0.0, "full", 1, 8).is_err());
        assert!(qjl(f32::NAN, "full", 1, 8).is_err());
        assert!(qjl(0.5, "full", 1, 8).is_ok());
    }
}
