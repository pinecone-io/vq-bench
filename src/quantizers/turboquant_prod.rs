//! `turboquant_prod`: TurboQuant's product variant — the bulk in a (b-1)-bit Gaussian
//! codebook, one bit reallocated to a 1-bit QJL of the residual for an unbiased
//! inner-product estimate.

use anyhow::{bail, ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::{get, get_or};
use super::qjl::Qjl;
use crate::coding::CodeLayout;
use crate::{
    CastNormal, NormalScale, Normalize, Params, Pipeline, Primitive, Quantizer, RandomHadamard,
    RandomRotate,
};

/// Independent seed offset for the residual's QJL rotation.
const RESIDUAL_ROTATION_SEED: u64 = 0xD15C0;

/// The `turboquant_prod` family. `b` in `2..=MAX_BITS`: `b-1` bits for the plain
/// Gaussian codebook, one bit for a QJL of the residual (its own rotation, on the
/// post-rotation width).
pub struct TurboquantProd(pub Pipeline);

impl TurboquantProd {
    /// `Normalize -> rotate(seed) -> CastNormal(b-1, Plain)`, then a 1-bit QJL of the
    /// residual (same rotation kind, its own seed, at the post-rotation width) —
    /// composed via [`Qjl::pipeline`]; a `Pipeline` is a `Primitive`.
    pub fn pipeline(bits: u8, rotation: &str, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (2..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 2..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        let stage: Box<dyn Primitive> = match rotation {
            "full" => Box::new(RandomRotate::new(seed)),
            "hadamard" => Box::new(RandomHadamard::new(dim, seed)),
            other => bail!("unknown rotation `{other}` (expected `full` or `hadamard`)"),
        };
        let mid_dim = stage.out_dim(dim); // width the residual lives in (padded under Hadamard)
        let residual = Qjl::pipeline(1.0, rotation, seed ^ RESIDUAL_ROTATION_SEED, mid_dim)?;
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
}

impl Quantizer for TurboquantProd {
    fn name() -> &'static str {
        "turboquant_prod"
    }

    fn display_name() -> &'static str {
        "TurboQuant-prod"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Normalize -> Rotate -> CastNormal(b-1, plain) -> QJL"
    }

    fn build(p: &Params, seed: u64, dim: usize) -> Result<Self> {
        let rotation = get_or(p, "rotation", "hadamard".to_string())?;
        Ok(Self(Self::pipeline(get(p, "b")?, &rotation, seed, dim)?))
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

    /// The `turboquant_prod` quantizer, with `rotation` given as configs give it.
    fn turboquant_prod(bits: u8, rotation: &str, seed: u64, dim: usize) -> Result<TurboquantProd> {
        let p = params(&[("b", json!(bits)), ("rotation", json!(rotation))]);
        TurboquantProd::build(&p, seed, dim)
    }

    #[test]
    fn rejects_out_of_range_bits() {
        assert!(turboquant_prod(1, "full", 1, 8).is_err()); // b-1 would be 0
        assert!(turboquant_prod(9, "full", 1, 8).is_err());
        assert!(turboquant_prod(2, "full", 1, 8).is_ok());
        assert!(turboquant_prod(8, "full", 1, 8).is_ok());
    }

    /// The QJL residual bit makes the inner-product estimate unbiased: slope ~ 1.
    #[test]
    fn unbiased_score_slope() {
        let mut rng = math::seed(13);
        let d = 256;
        let v: Array2<f32> = math::gaussian(&mut rng, (80, d));
        let q: Array2<f32> = math::gaussian(&mut rng, (10, d));
        for rotation in ["full", "hadamard"] {
            let codec = turboquant_prod(4, rotation, 1, d).unwrap();
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
