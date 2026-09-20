//! PCA_ROTATE: rotates every vector onto the principal axes of the fit set
//! -
//! Fit: eigendecompose the second moment
//! Model: R, the eigenvectors as columns, largest variance first
//! Code for vector x: empty
//! Apply: x --> x R
//! Reconstruct: y --> R^T * y   (R^T = R^-1)
//! Score: s --> s  (queries also rotated)
//!
//! Assumes a centered fit set: compose `center` in front.

use ndarray::{Array2, ArrayView2};

use super::rotation_model;
use crate::{coding, math, Primitive};

pub struct PcaRotate;

impl Primitive for PcaRotate {
    fn describe() -> &'static str {
        "rotate onto the principal axes of the fit set, largest variance first"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let (_, axes) = math::symmetric_eigen(math::second_moment(vectors).view());
        coding::pack_model(axes)
    }

    // encode omitted: a rotation owns no per-vector bits.

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        rotation_model::rotate(model, vectors);
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        rotation_model::rotate(model, queries);
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        rotation_model::unrotate(model, child_recons.expect("PcaRotate is not terminal"))
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Queries are rotated the same way, so the child's scores pass through.
        child_scores.expect("PcaRotate is not terminal").to_owned()
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0) // no per-vector bits: the rotation lives in the model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores, with_variances};
    use crate::Kmeans;
    use ndarray::{Array2, Axis};

    /// Data with a prescribed spectrum: [`with_variances`], centered as the stage expects,
    /// then rotated so the axes are not already principal.
    fn with_spectrum(n: usize, variances: &[f32], seed: u64) -> Array2<f32> {
        let d = variances.len();
        let mut x = with_variances(n, variances, seed);
        let mean = x.mean_axis(Axis(0)).unwrap();
        x -= &mean.broadcast((n, d)).unwrap();
        math::matmul(x.view(), math::random_orthogonal(&mut math::seed(seed ^ 0xf00), d).view())
    }

    /// The rotated data's column variances are the prescribed spectrum, in descending
    /// order — the mixing rotation is undone.
    #[test]
    fn recovers_the_spectrum_in_descending_order() {
        let spectrum = [4.0f32, 2.0, 1.0, 0.25];
        let v = with_spectrum(20000, &spectrum, 1);
        let model = PcaRotate.fit(v.view(), None);
        let mut x = v.clone();
        PcaRotate.apply(&model, &mut x, &[]);
        let found = x.var_axis(Axis(0), 0.0);
        for (j, &want) in spectrum.iter().enumerate() {
            assert!((found[j] - want).abs() < 0.15, "axis {j}: {} vs {want}", found[j]);
        }
    }

    #[test]
    fn orthogonal_round_trip_and_dot() {
        let v = with_spectrum(200, &[4.0, 2.0, 1.0, 0.25], 2);
        let q = math::gaussian(&mut math::seed(3), (3, 4));
        let model = PcaRotate.fit(v.view(), None);
        let mut x = v.clone();
        PcaRotate.apply(&model, &mut x, &[]);
        assert_close(&PcaRotate.reconstruct(&model, &[], Some(x.view())), &v, 1e-3);
        let mut rq = q.clone();
        PcaRotate.apply_queries(&model, &mut rq);
        assert_close(&rq.dot(&x.t()), &q.dot(&v.t()), 1e-3);
    }

    #[test]
    fn composes_in_pipeline() {
        let v = with_spectrum(80, &[4.0, 2.0, 1.0, 0.5, 0.25, 0.2, 0.1, 0.05], 4);
        let q: Array2<f32> = math::gaussian(&mut math::seed(5), (5, 8));
        assert_pipeline_scores(
            vec![Box::new(PcaRotate) as Box<dyn Primitive>, Box::new(Kmeans::new(16, 3))],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
