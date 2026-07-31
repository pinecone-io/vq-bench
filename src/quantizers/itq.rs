//! `itq`: Iterative Quantization — center, unit-normalize, then a learned rotation
//! minimizing sign-quantization error

use anyhow::Result;
use ndarray::{Array2, ArrayView2};

use crate::{CastHamming, Center, Normalize, OptimizeSigns, Params, Pipeline, Primitive, Quantizer};

/// The `itq` family. It takes no params.
pub struct Itq(pub Pipeline);

impl Itq {
    /// `Center -> Normalize -> OptimizeSigns(seed) -> CastHamming` over input dim `dim`.
    pub fn pipeline(seed: u64, dim: usize) -> Result<Pipeline> {
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                Box::new(OptimizeSigns::new(seed)),
                Box::new(CastHamming),
            ],
        )
    }
}

impl Quantizer for Itq {
    fn name() -> &'static str {
        "itq"
    }

    fn display_name() -> &'static str {
        "ITQ"
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> OptimizeSigns -> CastHamming"
    }

    fn build(_p: &Params, seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(seed, dim)?))
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
    use ndarray::{s, Array2};

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// Each query is a noised copy of one base vector and must score it highest.
    #[test]
    fn recovers_nearest_neighbor() {
        let mut rng = math::seed(42);
        let v: Array2<f32> = math::gaussian(&mut rng, (40, 128));
        let q = &v.slice(s![0..8, ..]).to_owned() + &(0.3 * math::gaussian(&mut rng, (8, 128)));
        let codec = Itq::build(&params(&[]), 1, 128).unwrap();
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let est = codec.score(&model, q.view(), &refs(&codes));
        let mut hits = 0;
        for i in 0..8 {
            let row = est.row(i);
            let argmax = (0..row.len()).max_by(|&a, &b| row[a].total_cmp(&row[b])).unwrap();
            if argmax == i {
                hits += 1;
            }
        }
        assert!(hits >= 7, "top-1 hits {hits}/8");
    }
}
