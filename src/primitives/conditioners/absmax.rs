//! ABSMAX: scales each vector into [-1,1]^d by dividing by its maximum magnitude coordinate
//! -
//! Model: empty
//! Code for vector x: max_i |x_i|
//! Apply: x --> x / max_i |x_i|
//! Reconstruct: y --> max_i |x_i| * y
//! Score: s --> max_i |x_i| * s

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::coding::CodeLayout;
use crate::{math, Primitive};
pub struct AbsMax;

/// The code layout: one trailing scalar (the per-vector max-abs), no bit levels.
fn layout() -> CodeLayout {
    CodeLayout::new().scalars(1)
}

/// The per-vector scale carried in the codes.
fn scales(codes: &[&[u8]]) -> Array1<f32> {
    let (_, [scale]) = layout().unpack::<1>(codes);
    scale
}

impl Primitive for AbsMax {
    fn describe() -> &'static str {
        "scale each vector into [-1,1] by dividing by max absolute value"
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let scales = vectors
            .mapv(f32::abs)
            .fold_axis(Axis(1), 0.0f32, |&a, &b| a.max(b));
        layout().pack_scalars(&[scales.view()])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        math::scale_rows(vectors, math::reciprocal(scales(codes).view()).view());
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("AbsMax is not terminal").to_owned();
        math::scale_rows(&mut out, scales(codes).view());
        out
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_scores.expect("AbsMax is not terminal").to_owned();
        math::scale_cols(&mut out, scales(codes).view());
        out
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(layout().byte_len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores, refs};
    use ndarray::array;

    #[test]
    fn max_abs_then_restore() {
        let v = array![[3., -4., 2.], [0., 5., -1.], [1., 1., 1.]];
        let am = AbsMax;
        let codes = am.encode(&[], v.view());
        let r = refs(&codes);
        let mut x = v.clone();
        am.apply(&[], &mut x, &r);
        for &val in x.iter() {
            assert!((-1.0..=1.0).contains(&val), "coord {val} out of [-1,1]");
        }
        assert_close(&am.reconstruct(&[], &r, Some(x.view())), &v, 1e-4);
    }

    #[test]
    fn score_recovers_dot() {
        let v = array![[3., -4., 0.], [1., 2., 2.], [-1., 0., 2.]];
        let q = array![[1., 0., -1.], [0.5, 1., 0.]];
        let am = AbsMax;
        let codes = am.encode(&[], v.view());
        let r = refs(&codes);
        let mut xhat = v.clone();
        am.apply(&[], &mut xhat, &r);
        let child = q.dot(&xhat.t()); // <q, xhat>
        assert_close(
            &am.score(&[], q.view(), &r, Some(child.view())),
            &q.dot(&v.t()),
            1e-3,
        );
    }

    #[test]
    fn composes_in_pipeline() {
        // A lossless conditioner stack: confirm only the score invariant against the
        // pipeline's own reconstruction (reconstruction accuracy is minmax/cast's job).
        let v = array![[0., 1., 2., 3.], [4., -6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        assert_pipeline_scores(
            vec![
                Box::new(AbsMax) as Box<dyn Primitive>,
                Box::new(crate::MinMax::default()),
                Box::new(crate::CastUint::new(8)),
            ],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
