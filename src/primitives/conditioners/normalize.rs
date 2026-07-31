//! NORMALIZE: scales each vector to unit norm, carrying ||x|| as side info
//! -
//! Model: empty
//! Code for vector x: ||x||
//! Apply: x --> x / ||x||
//! Reconstruct: y --> ||x|| * y
//! Score: s --> ||x|| * s

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::coding::CodeLayout;
use crate::{math, Primitive};

pub struct Normalize;

/// The code layout: one trailing scalar (the per-vector L2 norm), no bit levels.
fn layout() -> CodeLayout {
    CodeLayout::new().scalars(1)
}

/// The per-vector norm carried in the codes.
fn norms(codes: &[&[u8]]) -> Array1<f32> {
    let (_, [norm]) = layout().unpack::<1>(codes);
    norm
}

impl Primitive for Normalize {
    fn describe() -> &'static str {
        "scale each vector to unit L2 norm"
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let norms = vectors.mapv(|x| x * x).sum_axis(Axis(1)).mapv(f32::sqrt);
        layout().pack_scalars(&[norms.view()])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        math::scale_rows(vectors, math::reciprocal(norms(codes).view()).view());
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("Normalize is not terminal").to_owned();
        math::scale_rows(&mut out, norms(codes).view());
        out
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_scores.expect("Normalize is not terminal").to_owned();
        math::scale_cols(&mut out, norms(codes).view());
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
    fn unit_norm_then_restore() {
        let v = array![[3., 4.], [0., 5.], [1., 1.]];
        let nz = Normalize;
        let codes = nz.encode(&[], v.view());
        let r = refs(&codes);
        let mut x = v.clone();
        nz.apply(&[], &mut x, &r);
        // each row is unit norm after apply
        for row in x.rows() {
            let norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "norm {norm}");
        }
        assert_close(&nz.reconstruct(&[], &r, Some(x.view())), &v, 1e-4);
    }

    #[test]
    fn score_recovers_dot() {
        // normalize on top of an (identity) lossless child: ||x|| * <q, xhat> = <q, x>.
        let v = array![[3., 4., 0.], [1., 2., 2.], [-1., 0., 2.]];
        let q = array![[1., 0., -1.], [0.5, 1., 0.]];
        let nz = Normalize;
        let codes = nz.encode(&[], v.view());
        let r = refs(&codes);
        let mut xhat = v.clone();
        nz.apply(&[], &mut xhat, &r);
        let child = q.dot(&xhat.t()); // <q, xhat>
        assert_close(
            &nz.score(&[], q.view(), &r, Some(child.view())),
            &q.dot(&v.t()),
            1e-3,
        );
    }

    #[test]
    fn composes_in_pipeline() {
        // normalize -> minmax -> cast: a lossless conditioner stacked on the existing pair.
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        assert_pipeline_scores(
            vec![
                Box::new(Normalize) as Box<dyn Primitive>,
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
