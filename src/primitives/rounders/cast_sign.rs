//! CAST(SIGN): one sign bit per coordinate, scored as the plain asymmetric dot
//! -
//! Model: input dim d
//! Code for vector x: one sign bit per coordinate (1 iff x_i >= 0), decoded to +-1
//! Apply: x --> x - sign(x)              (residual for the next stage)
//! Reconstruct: y --> sign(x) + y
//! Score: s --> <q, sign(x)> + s

use ndarray::{Array2, ArrayView2};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

pub struct CastSign;

/// One sign bit per coordinate.
const BITS: u8 = 1;

/// The code layout: `d` sign bits, no scalars.
fn layout(d: usize) -> CodeLayout {
    CodeLayout::new().bits(d, BITS)
}

/// Decode packed sign bits to +-1 reals (d values per code).
fn decode_signs(codes: &[&[u8]], d: usize) -> Array2<f32> {
    let (levels, []) = layout(d).unpack::<0>(codes);
    levels.mapv(|b| if b == 1 { 1.0 } else { -1.0 })
}

impl Primitive for CastSign {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let signs = vectors.mapv(|x| (x >= 0.0) as u32);
        layout(vectors.ncols()).pack(signs.view(), &[])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        *vectors -= &decode_signs(codes, vectors.ncols());
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let mut out = decode_signs(codes, d);
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
        let mut out = math::matmul(queries, decode_signs(codes, queries.ncols()).t());
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, in_dim: usize) -> Option<usize> {
        Some(layout(in_dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use ndarray::array;

    #[test]
    fn round_trip_and_score_exact_on_sign_grid() {
        let v = array![[1., -1., 1., 1.], [-1., -1., 1., -1.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let cast = CastSign;
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(&cast.reconstruct(&model, &r, None), &v, 1e-6);
        assert_close(&cast.score(&model, q.view(), &r, None), &q.dot(&v.t()), 1e-4);
    }

    #[test]
    fn encodes_sign_bit_zero_is_positive() {
        let v = array![[0.0, -0.2, 3.0, -5.0]];
        let cast = CastSign;
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let recon = cast.reconstruct(&model, &refs(&codes), None);
        assert_close(&recon, &array![[1., -1., 1., -1.]], 1e-6);
    }

    #[test]
    fn code_bytes_matches_emitted() {
        let v = array![[0.1, -0.2, 0.3, 0.4, -0.5]]; // d = 5
        let cast = CastSign;
        let codes = cast.encode(&cast.fit(v.view(), None), v.view());
        assert_eq!(codes[0].len(), cast.code_bytes(5).unwrap());
        assert_eq!(codes[0].len(), 5usize.div_ceil(8)); // 1 byte
    }

    #[test]
    fn composes_after_absmax() {
        use crate::{AsQuantizer, Pipeline, Quantizer};
        let v = array![[2., -2., 2., 2.], [-4., -4., 4., -4.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(
            Pipeline::new(4, vec![Box::new(crate::AbsMax) as Box<dyn Primitive>, Box::new(CastSign)]).unwrap(),
        );
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(&codec.reconstruct(&model, &r), &v, 1e-5);
        assert_close(&codec.score(&model, q.view(), &r), &q.dot(&v.t()), 1e-4);
    }
}
