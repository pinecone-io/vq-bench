//! CAST(HAMMING): one sign bit per coordinate, scored by a SimHash angle estimate
//! -
//! Model: input dim d
//! Code for vector x: one sign bit per coordinate (1 iff x_i >= 0)
//! Apply: x --> x - sign(x)/sqrt(d)              (residual for the next stage)
//! Reconstruct: y --> sign(x)/sqrt(d) + y
//! Score: s --> ||q|| * cos(pi * hamming / d) + s, hamming = (d - <sign(q), b>)/2

use ndarray::{Array2, ArrayView2, Axis};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

pub struct CastHamming;

const BITS: u8 = 1;

fn layout(d: usize) -> CodeLayout {
    CodeLayout::new().bits(d, BITS)
}

fn decode_signs(codes: &[&[u8]], d: usize) -> Array2<f32> {
    let (levels, []) = layout(d).unpack::<0>(codes);
    levels.mapv(|b| if b == 1 { 1.0 } else { -1.0 })
}

/// Decode signs to the unit-norm direction +-1/sqrt(d).
fn decode_unit(codes: &[&[u8]], d: usize) -> Array2<f32> {
    decode_signs(codes, d).mapv(|s| s / (d as f32).sqrt())
}

impl Primitive for CastHamming {
    fn describe() -> &'static str {
        "round vector and query to ±1"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.ncols())
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let signs = vectors.mapv(|x| (x >= 0.0) as u32);
        layout(vectors.ncols()).pack(signs.view(), &[])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        *vectors -= &decode_unit(codes, vectors.ncols());
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let d = super::code_dim(model, child_recons);
        let mut out = decode_unit(codes, d);
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
        let d = queries.ncols();
        let d_f32 = d as f32;
        let query_signs = queries.mapv(|x| if x >= 0.0 { 1.0 } else { -1.0 });
        let dots = math::matmul(query_signs.view(), decode_signs(codes, d).t()); // <sign(q), b>
        let mut out = dots.mapv(|dot| (std::f32::consts::PI * (d_f32 - dot) * 0.5 / d_f32).cos());
        let qnorm = queries.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
        math::scale_rows(&mut out, qnorm.view());
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, _model: &[u8], in_dim: usize) -> Option<usize> {
        Some(layout(in_dim).byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use ndarray::array;

    #[test]
    fn reconstruct_is_unit_signs() {
        let v = array![[0.5, -0.2, 3.0, -1.0]];
        let cast = CastHamming;
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let recon = cast.reconstruct(&model, &refs(&codes), None);
        let s = 1.0 / 4f32.sqrt();
        assert_close(&recon, &array![[s, -s, s, -s]], 1e-6);
    }

    #[test]
    fn score_matches_simhash_formula() {
        let v = array![[1., 1., -1., 1.]];
        let cast = CastHamming;
        let model = cast.fit(v.view(), None);
        let codes = cast.encode(&model, v.view());
        let r = refs(&codes);
        let q_same = array![[2., 3., -1., 4.]];
        let q_opp = array![[-2., -3., 1., -4.]];
        let norm = (2f32 * 2. + 3. * 3. + 1. + 4. * 4.).sqrt();
        let s_same = cast.score(&model, q_same.view(), &r, None);
        let s_opp = cast.score(&model, q_opp.view(), &r, None);
        assert!((s_same[[0, 0]] - norm).abs() < 1e-4, "{}", s_same[[0, 0]]);
        assert!((s_opp[[0, 0]] + norm).abs() < 1e-4, "{}", s_opp[[0, 0]]);
    }

    #[test]
    fn code_bytes_matches_emitted() {
        let v = array![[0.1, -0.2, 0.3, 0.4, -0.5]]; // d = 5
        let cast = CastHamming;
        let codes = cast.encode(&cast.fit(v.view(), None), v.view());
        assert_eq!(codes[0].len(), cast.code_bytes(&[], 5).unwrap());
        assert_eq!(codes[0].len(), 5usize.div_ceil(8));
    }

    #[test]
    fn composes_after_normalize() {
        use crate::{AsQuantizer, Pipeline, Quantizer};
        let v = array![[0., 1., 2., 3.], [4., -6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(
            Pipeline::new(4, vec![Box::new(crate::Normalize) as Box<dyn Primitive>, Box::new(CastHamming)]).unwrap(),
        );
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        assert_eq!(codec.score(&model, q.view(), &r).dim(), (2, 3));
    }
}
