//! `pq`: Product Quantization — split into `section_dim`-column segments, each
//! quantized by its own `centroids`-codeword k-means codebook.

use anyhow::{ensure, Result};

use super::catalog::{get, QuantizerSpec};
use crate::coding::CodeLayout;
use crate::{Kmeans, Pipeline, Primitive, SegmentSplit, Split, Splitter};

pub const SPEC: QuantizerSpec = QuantizerSpec {
    key: "pq",
    family: "PQ",
    params: &["centroids", "section_dim"],
    describe: "SegmentSplit(section_dim) -> [Kmeans(centroids)]",
    build: |p, seed, dim| pq(get(p, "centroids")?, get(p, "section_dim")?, seed, dim),
};

/// Contiguous `section_dim`-column segments (dimension-independent), each rounded to
/// its own `centroids`-codeword k-means codebook (distinct seed per segment).
pub fn pq(centroids: usize, section_dim: usize, seed: u64, dim: usize) -> Result<Pipeline> {
    ensure!(
        (2..=1 << CodeLayout::MAX_BITS).contains(&centroids),
        "centroids must be in 2..={}, got {centroids}",
        1u32 << CodeLayout::MAX_BITS
    );
    ensure!(
        (1..=dim).contains(&section_dim),
        "section_dim must be in 1..={dim}, got {section_dim}"
    );
    let split = SegmentSplit::new(dim, section_dim);
    let children = (0..split.n_branches())
        .map(|b| {
            Pipeline::new(
                split.branch_in_dim(&[], dim, b),
                vec![Box::new(Kmeans::new(centroids, seed.wrapping_add(b as u64))) as Box<dyn Primitive>],
            )
        })
        .collect::<Result<Vec<_>>>()?;
    Pipeline::new(dim, vec![Box::new(Split::new(split, children))])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{math, AsQuantizer, Quantizer};
    use ndarray::Array2;

    #[test]
    fn rejects_bad_params() {
        assert!(pq(1, 8, 1, 64).is_err()); // < 2 centroids
        assert!(pq(257, 8, 1, 64).is_err()); // > 256 centroids
        assert!(pq(16, 0, 1, 64).is_err()); // section_dim < 1
        assert!(pq(16, 65, 1, 64).is_err()); // section_dim > dim
        assert!(pq(16, 8, 1, 64).is_ok());
    }

    /// PQ scores exactly against its own (lossy) reconstruction: per-segment ADC, summed.
    #[test]
    fn score_matches_reconstruction() {
        let v: Array2<f32> = math::gaussian(&mut math::seed(1), (60, 32));
        let q: Array2<f32> = math::gaussian(&mut math::seed(2), (8, 32));
        let codec = AsQuantizer(pq(16, 8, 1, 32).unwrap()); // 8-dim segments (4 of them), k=16
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        let recon = codec.reconstruct(&model, &r);
        assert_close(&codec.score(&model, q.view(), &r), &q.dot(&recon.t()), 1e-3);
    }
}
