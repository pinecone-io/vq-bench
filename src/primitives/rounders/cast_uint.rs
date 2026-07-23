//! CAST(UINT, b): rounds [0, 1] input into a b-bit uniform lattice of N = 2^b bins
//! -
//! Model: input dim d
//! Code for vector x: b-bit bin index per coordinate, bin q = floor(x * N)
//! Apply: x --> x - center(q)              (residual for the next stage)
//! Reconstruct: y --> center(q) + y
//! Score: s --> (<q, c> + 0.5 * sum(q)) / N + s   (from the bin indices c, not values)
//!
//! Bin q covers [q/N, (q+1)/N) and decodes to its center (q + 0.5)/N.

use ndarray::{Array2, ArrayView2, Axis};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

pub struct CastUint {
    bits: u8,
}

impl CastUint {
    /// Cast to `bits`-bit unsigned; the valid range (`1..=CodeLayout::MAX_BITS`) is
    /// enforced by the quantizer builder.
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }

    /// Number of uniform bins `N = 2^bits`.
    fn bins(&self) -> u32 {
        1u32 << self.bits
    }

    /// The code layout: `d` levels of `self.bits` bits, no scalars.
    fn layout(&self, d: usize) -> CodeLayout {
        CodeLayout::new().bits(d, self.bits)
    }

    /// Decode `n` codes of `d` values to their bin centers `(bin + 0.5)/N` in `[0, 1]`.
    fn decode(&self, codes: &[&[u8]], d: usize) -> Array2<f32> {
        let inv_bins = 1.0 / self.bins() as f32;
        let (levels, []) = self.layout(d).unpack::<0>(codes);
        levels.mapv(|bin| (bin as f32 + 0.5) * inv_bins)
    }
}

impl Primitive for CastUint {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let bins = self.bins();
        let bins_f = bins as f32;
        // Bin index floor(x * N) per coordinate (x = 1.0 folds into the top bin).
        let levels = vectors.mapv(|x| ((x.clamp(0.0, 1.0) * bins_f) as u32).min(bins - 1));
        self.layout(vectors.ncols()).pack(levels.view(), &[])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        *vectors -= &self.decode(codes, vectors.ncols());
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let mut out = self.decode(codes, d);
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Score straight from the integer bin indices
        // <q, center(c)> = (<q, c> + 0.5 * sum(q)) / N.
        let d = queries.ncols();
        let inv_bins = 1.0 / self.bins() as f32;
        let (levels, []) = self.layout(d).unpack::<0>(codes);
        let levels = levels.mapv(|bin| bin as f32);
        let mut out = math::matmul(queries, levels.t()); // <q, c>
        math::offset_rows(&mut out, queries.sum_axis(Axis(1)).mapv(|sum| 0.5 * sum).view()); // + 0.5 * sum(q)
        out *= inv_bins; // / N
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, in_dim: usize) -> Option<usize> {
        Some(self.layout(in_dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use ndarray::array;

    #[test]
    fn round_trip_within_lattice_error() {
        let v = array![[0.0, 0.2, 0.55, 1.0], [0.9, 0.1, 0.33, 0.67]];
        let cast = CastUint::new(8);
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        // Max error is a half-bin = 1/(2*2^8) = 1/512.
        assert_close(
            &cast.reconstruct(&model, &refs(&codes), None),
            &v,
            1.0 / 512.0 + 1e-6,
        );
    }

    #[test]
    fn exact_at_bin_centers() {
        // Bin centers for bits=2 (N=4): 1/8, 3/8, 5/8, 7/8 -- reconstruct exactly.
        let v = array![
            [1. / 8., 3. / 8., 5. / 8., 7. / 8.],
            [7. / 8., 1. / 8., 3. / 8., 5. / 8.]
        ];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let cast = CastUint::new(2);
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(&cast.reconstruct(&model, &r, None), &v, 1e-6);
        assert_close(
            &cast.score(&model, q.view(), &r, None),
            &q.dot(&v.t()),
            1e-4,
        );
    }

    #[test]
    fn code_bytes_matches_emitted() {
        let v = array![[0.1, 0.2, 0.3, 0.4, 0.5]]; // d = 5
        for bits in [1u8, 2, 4, 5, 8] {
            let cast = CastUint::new(bits);
            let codes = cast.encode(&cast.fit(v.view(), None), v.view());
            assert_eq!(codes[0].len(), cast.code_bytes(5).unwrap());
            assert_eq!(codes[0].len(), (5 * bits as usize).div_ceil(8));
        }
    }

    #[test]
    fn size_accounting() {
        use crate::{byte_split, AsQuantizer, Pipeline, Quantizer};
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.]]; // 2 vectors, d = 4
        let codec = AsQuantizer(
            Pipeline::new(
                4,
                vec![
                    Box::new(crate::MinMax::default()) as Box<dyn Primitive>,
                    Box::new(CastUint::new(2)),
                ],
            )
            .unwrap(),
        );
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        // per vector: minmax 8 bytes + cast ceil(4*2/8)=1 = 9; x2 = 18.
        assert_eq!(byte_split(&model, &codes).1, 18);
    }
}
