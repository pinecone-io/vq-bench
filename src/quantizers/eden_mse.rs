//! `eden_mse`: unit-normalize, rotate, then a b-bit Gaussian codebook with the
//! MSE-optimal dequant scale (EDEN).

use anyhow::{bail, ensure, Result};
use ndarray::{Array2, ArrayView2};

use super::catalog::{get, get_or};
use crate::coding::CodeLayout;
use crate::{
    CastNormal, NormalScale, Normalize, Params, Pipeline, Primitive, Quantizer, RandomHadamard,
    RandomRotate,
};

/// The `eden_mse` family. `get`/`get_or` type-check the params; the `b` range is
/// checked in `build`.
pub struct EdenMse(pub Pipeline);

impl EdenMse {
    /// `Normalize -> rotate(seed) -> CastNormal(b, BiasedMse)` over input dim `dim`.
    pub fn pipeline(bits: u8, rotation: &str, seed: u64, dim: usize) -> Result<Pipeline> {
        ensure!(
            (1..=CodeLayout::MAX_BITS).contains(&bits),
            "b must be in 1..={}, got {bits}",
            CodeLayout::MAX_BITS
        );
        let rotation: Box<dyn Primitive> = match rotation {
            "full" => Box::new(RandomRotate::new(seed)),
            "hadamard" => Box::new(RandomHadamard::new(dim, seed)),
            other => bail!("unknown rotation `{other}` (expected `full` or `hadamard`)"),
        };
        Pipeline::new(
            dim,
            vec![
                Box::new(Normalize),
                rotation,
                Box::new(CastNormal::new(bits, NormalScale::BiasedMse)),
            ],
        )
    }
}

impl Quantizer for EdenMse {
    fn name() -> &'static str {
        "eden_mse"
    }

    fn display_name() -> &'static str {
        "EDEN-MSE"
    }

    fn params() -> &'static [&'static str] {
        &["b", "rotation"]
    }

    fn describe() -> &'static str {
        "Normalize -> Rotate -> CastNormal(b, MSE)"
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

    /// The `eden_mse` quantizer, with `rotation` given as configs give it.
    fn eden_mse(bits: u8, rotation: &str, seed: u64, dim: usize) -> Result<EdenMse> {
        let p = params(&[("b", json!(bits)), ("rotation", json!(rotation))]);
        EdenMse::build(&p, seed, dim)
    }

    /// `b` out of `1..=8` or an unknown rotation name is a build error (surfaced by
    /// `RunConfig::validate`).
    #[test]
    fn rejects_bad_params() {
        assert!(eden_mse(0, "full", 1, 8).is_err());
        assert!(eden_mse(9, "full", 1, 8).is_err());
        assert!(eden_mse(4, "diagonal", 1, 8).is_err());
        assert!(eden_mse(1, "full", 1, 8).is_ok());
        assert!(eden_mse(8, "full", 1, 8).is_ok());
    }

    /// The asymmetric score equals the exact dot with the pipeline's own
    /// reconstruction, under both rotations.
    #[test]
    fn score_tracks_reconstruction() {
        let mut rng = math::seed(2);
        let v: Array2<f32> = math::gaussian(&mut rng, (40, 128));
        let q: Array2<f32> = math::gaussian(&mut rng, (6, 128));
        for rotation in ["full", "hadamard"] {
            let codec = eden_mse(6, rotation, 1, 128).unwrap();
            let model = codec.fit(v.view(), None);
            let codes = codec.encode(&model, v.view());
            let r = refs(&codes);
            let recon = codec.reconstruct(&model, &r);
            let est = codec.score(&model, q.view(), &r);
            let exact = q.dot(&recon.t());
            for (a, b) in est.iter().zip(exact.iter()) {
                assert!((a - b).abs() < 1e-2, "score {a} vs {b}");
            }
        }
    }
}
