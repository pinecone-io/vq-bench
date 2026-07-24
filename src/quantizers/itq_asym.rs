//! `itq_asym`: asymmetric ITQ — center, unit-normalize, then a learned sign-
//! quantization rotation, scored by the continuous query against the binary codes.

use anyhow::Result;

use super::catalog::QuantizerSpec;
use crate::{CastSign, Center, Normalize, OptimizeSigns, Pipeline};

pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "itq_asym",
    family: "ITQ-asym",
    params: &[],
    describe: "Center -> Normalize -> OptimizeSigns -> CastSign",
    build: |_p, seed, dim| itq_asym(seed, dim),
};

pub fn itq_asym(seed: u64, dim: usize) -> Result<Pipeline> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{math, AsQuantizer, Quantizer};
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
        let codec = AsQuantizer(itq_asym(1, 128).unwrap());
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
