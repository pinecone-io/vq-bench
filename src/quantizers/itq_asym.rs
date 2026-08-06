//! `itq_asym`: asymmetric ITQ — center, unit-normalize, then a learned sign-
//! quantization rotation, scored by the continuous query against the binary codes.

use anyhow::Result;

use crate::{CastSign, Center, Normalize, OptimizeSigns, Params, Pipeline, Quantizer};

/// The `itq_asym` family. It takes no params.
pub struct ItqAsym(pub Pipeline);

impl ItqAsym {
    /// `Center -> Normalize -> OptimizeSigns(seed) -> CastSign` over input dim `dim`.
    pub fn pipeline(seed: u64, dim: usize) -> Result<Pipeline> {
        Pipeline::new(
            dim,
            vec![
                Box::new(Center),
                Box::new(Normalize),
                Box::new(OptimizeSigns::new(seed)),
                Box::new(CastSign),
            ],
        )
    }
}

impl Quantizer for ItqAsym {
    fn name() -> &'static str {
        "itq_asym"
    }

    fn display_name() -> &'static str {
        "ITQ-asym"
    }

    fn describe() -> &'static str {
        "Center -> Normalize -> OptimizeSigns -> CastSign"
    }

    fn build(_p: &Params, seed: u64, dim: usize) -> Result<Self> {
        Ok(Self(Self::pipeline(seed, dim)?))
    }

    crate::pipeline_quantizer!();
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
        let codec = ItqAsym::build(&params(&[]), 1, 128).unwrap();
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
