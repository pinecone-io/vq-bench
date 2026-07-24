//! `simhash`: center, unit-normalize, rotate, then 1-bit signs scored by a
//! Hamming-angle inner-product estimate.

use anyhow::Result;

use super::catalog::{get_or, QuantizerSpec};
use super::Rotation;
use crate::{CastHamming, Center, Normalize, Pipeline};

pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "simhash",
    family: "SimHash",
    params: &["rotation"],
    describe: "Center -> Normalize -> Rotate -> CastHamming",
    build: |p, seed, dim| simhash(get_or(p, "rotation", Rotation::Full)?, seed, dim),
};

pub fn simhash(rotation: Rotation, seed: u64, dim: usize) -> Result<Pipeline> {
    Pipeline::new(
        dim,
        vec![
            Box::new(Center),
            Box::new(Normalize),
            rotation.stage(dim, seed),
            Box::new(CastHamming),
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
        for rotation in [Rotation::Full, Rotation::Hadamard] {
            let codec = AsQuantizer(simhash(rotation, 1, 128).unwrap());
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
}
