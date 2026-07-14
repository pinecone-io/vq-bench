//! `minmax`: per-vector rescale to `[0, 1]`, then a `b`-bit uniform lattice.

use crate::{catalog, CastUint, MinMax, NamedQuantizer, Pipeline};

/// `minmax` to `[0, 1]` then `cast(uint, bits)` — `MinMax` feeds `CastUint` its
/// expected input range. Family name `MinMax`; `bits` (config key `b`) is a parameter.
pub fn minmax(bits: u8) -> NamedQuantizer {
    NamedQuantizer {
        name: catalog::display("minmax").to_string(),
        pipeline: Pipeline::new(vec![
            Box::new(MinMax::default()),
            Box::new(CastUint::new(bits)),
        ]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Quantizer;
    use ndarray::{array, Array2};

    fn refs(codes: &[Vec<u8>]) -> Vec<&[u8]> {
        codes.iter().map(Vec::as_slice).collect()
    }

    /// 8-bit round-trip recovers the input within one lattice step, and `score`
    /// tracks the exact dot product to the same tolerance.
    #[test]
    fn roundtrip_and_score() {
        // No constant rows: a zero-range vector is unrecoverable by `MinMax`.
        let v: Array2<f32> = array![[0., 1., 2.], [4., 2., 0.], [-1., 3., 1.], [2., 0., 3.]];
        let q: Array2<f32> = array![[1., 0., -1.], [0.5, 1., 2.]];
        let codec = minmax(8);
        assert_eq!(codec.name, "MinMax");

        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());

        let recon = codec.reconstruct(&model, &refs(&codes));
        for (x, y) in recon.iter().zip(v.iter()) {
            assert!((x - y).abs() < 0.05, "recon {x} vs {y}");
        }

        let est = codec.score(&model, q.view(), &refs(&codes));
        let exact = q.dot(&v.t());
        for (x, y) in est.iter().zip(exact.iter()) {
            assert!((x - y).abs() < 0.2, "score {x} vs {y}");
        }
    }
}
