//! `cast(uint, b)`: round `[0, 1]` input into a `b`-bit uniform lattice.

use ndarray::{Array2, ArrayView2};

use super::cast_common::{checked_dim, dim_bytes, stored_dim, MAX_BITS};
use crate::{coding, math, Primitive};

/// Quantize `[0, 1]` input into `2^bits` uniform bins, reconstructing each to its
/// **bin center**: bin `q` covers `[q/N, (q+1)/N)` and decodes to `(q+0.5)/N`
/// (`N = 2^bits`). E.g. `bits=2`: bins split at `1/4, 1/2, 3/4`, centers
/// `1/8, 3/8, 5/8, 7/8`. The model stores only the input dim `d` (so a terminal
/// `reconstruct` knows how many values a packed code holds).
pub struct CastUint {
    bits: u8,
}

impl CastUint {
    /// Cast to `bits`-bit unsigned (`1..=8`).
    pub fn new(bits: u8) -> Self {
        debug_assert!((1..=MAX_BITS).contains(&bits));
        Self { bits }
    }

    /// Number of uniform bins `N = 2^bits`.
    fn bins(&self) -> u32 {
        1u32 << self.bits
    }

    /// Decode `n` codes of `d` values to their bin centers in `[0, 1]`.
    fn decode(&self, codes: &[&[u8]], d: usize) -> Array2<f32> {
        let n = self.bins() as f32;
        coding::unpack_codes(codes, d, self.bits).mapv(|q| (q as f32 + 0.5) / n)
    }
}

impl Primitive for CastUint {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        dim_bytes(vectors.ncols())
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let d = checked_dim(model, vectors.ncols());
        let n = self.bins();
        let nf = n as f32;
        // Stream row by row: bin index ⌊x·N⌋ (x=1.0 folds into the top bin) into a
        // reused buffer, packed straight to that vector's code — no (n×d) intermediate.
        let mut levels = vec![0u32; d];
        vectors
            .rows()
            .into_iter()
            .map(|row| {
                for (q, &x) in levels.iter_mut().zip(row) {
                    *q = ((x.clamp(0.0, 1.0) * nf) as u32).min(n - 1);
                }
                coding::pack_bits(&levels, self.bits)
            })
            .collect()
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let d = checked_dim(model, vectors.ncols());
        *vectors -= &self.decode(codes, d); // residual
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = match child_recons {
            Some(c) => checked_dim(model, c.ncols()),
            None => stored_dim(model),
        };
        let mut out = self.decode(codes, d);
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Dot continuous queries against each candidate's OWN decoded lattice code
        // (not a pipeline reconstruct).
        let d = checked_dim(model, queries.ncols());
        let cand = self.decode(codes, d);
        let mut out = math::matmul(queries, cand.t());
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, in_dim: usize) -> Option<usize> {
        Some((in_dim * self.bits as usize).div_ceil(8))
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
        // Max error is a half-bin = 1/(2·2^8) = 1/512.
        assert_close(
            &cast.reconstruct(&model, &refs(&codes), None),
            &v,
            1.0 / 512.0 + 1e-6,
        );
    }

    #[test]
    fn exact_at_bin_centers() {
        // Bin centers for bits=2 (N=4): 1/8, 3/8, 5/8, 7/8 — reconstruct exactly.
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
        // per vector: minmax 8 bytes + cast ceil(4*2/8)=1 = 9; ×2 = 18.
        assert_eq!(byte_split(&model, &codes).1, 18);
    }
}
