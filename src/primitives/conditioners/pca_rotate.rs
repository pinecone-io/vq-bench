//! PCA_ROTATE: rotates every vector onto the principal axes of the fit set, keeping
//! the top `rank` of them
//! -
//! Fit: eigendecompose the second moment
//! Model: R (d x k), the top-k eigenvectors as columns, largest variance first
//! Code for vector x: empty
//! Apply: x --> x R   (d dims --> k dims)
//! Reconstruct: y --> y R^T   (R^T = R^-1 at k = d; the projection onto the kept axes below it)
//! Score: s --> s  (queries also rotated)
//!
//! Assumes a centered fit set: compose `center` in front. Without a rank, k = d.

use ndarray::{s, Array2, ArrayView2};

use super::rotation_model;
use crate::{coding, math, Primitive};

#[derive(Default)]
pub struct PcaRotate {
    rank: Option<usize>,
}

impl PcaRotate {
    /// Keep only the top `rank` principal axes: the output has `rank` dims.
    pub fn truncated(rank: usize) -> Self {
        assert!(rank > 0, "rank must be positive");
        Self { rank: Some(rank) }
    }
}

impl Primitive for PcaRotate {
    fn describe() -> &'static str {
        "rotate onto the principal axes of the fit set, largest variance first"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let d = vectors.ncols();
        let (_, axes) = math::symmetric_eigen(math::second_moment(vectors).view());
        let kept = self.out_dim(d);
        assert!(kept <= d, "rank {kept} exceeds the input dim {d}");
        coding::pack_model(axes.slice(s![.., ..kept]).to_owned())
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

    fn out_dim(&self, in_dim: usize) -> usize {
        self.rank.unwrap_or(in_dim)
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
        let pca = PcaRotate::default();
        let model = pca.fit(v.view(), None);
        let mut x = v.clone();
        pca.apply(&model, &mut x, &[]);
        let found = x.var_axis(Axis(0), 0.0);
        for (j, &want) in spectrum.iter().enumerate() {
            assert!((found[j] - want).abs() < 0.15, "axis {j}: {} vs {want}", found[j]);
        }
    }

    #[test]
    fn orthogonal_round_trip_and_dot() {
        let v = with_spectrum(200, &[4.0, 2.0, 1.0, 0.25], 2);
        let q = math::gaussian(&mut math::seed(3), (3, 4));
        let pca = PcaRotate::default();
        let model = pca.fit(v.view(), None);
        let mut x = v.clone();
        pca.apply(&model, &mut x, &[]);
        assert_close(&pca.reconstruct(&model, &[], Some(x.view())), &v, 1e-3);
        let mut rq = q.clone();
        pca.apply_queries(&model, &mut rq);
        assert_close(&rq.dot(&x.t()), &q.dot(&v.t()), 1e-3);
    }

    #[test]
    fn composes_in_pipeline() {
        let v = with_spectrum(80, &[4.0, 2.0, 1.0, 0.5, 0.25, 0.2, 0.1, 0.05], 4);
        let q: Array2<f32> = math::gaussian(&mut math::seed(5), (5, 8));
        assert_pipeline_scores(
            vec![
                Box::new(PcaRotate::default()) as Box<dyn Primitive>,
                Box::new(Kmeans::new(16, 3)),
            ],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }

    /// With a rank, apply lands in `rank` dims and reconstruct is the projection onto the
    /// top-`rank` principal axes, `x V_k V_k^T`; scores still pass through exactly.
    #[test]
    fn truncation_projects_onto_the_top_axes() {
        let (d, k) = (6, 3);
        let v = with_spectrum(300, &[8.0, 4.0, 2.0, 0.5, 0.25, 0.1], 6);
        let q = math::gaussian(&mut math::seed(7), (3, d));
        let pca = PcaRotate::truncated(k);
        assert_eq!(pca.out_dim(d), k);
        let model = pca.fit(v.view(), None);
        let mut y = v.clone();
        pca.apply(&model, &mut y, &[]);
        assert_eq!(y.dim(), (300, k));

        let recon = pca.reconstruct(&model, &[], Some(y.view()));
        assert_eq!(recon.dim(), (300, d));
        let full = rotation_model::matrix(&PcaRotate::default().fit(v.view(), None));
        let top = full.slice(s![.., ..k]);
        let projection = math::matmul(v.view(), math::matmul(top, top.t()).view());
        assert_close(&recon, &projection, 1e-3);

        let mut rq = q.clone();
        pca.apply_queries(&model, &mut rq);
        assert_close(&rq.dot(&y.t()), &q.dot(&recon.t()), 1e-3);
    }

    /// `rank = dim` is the square rotation: same model bytes, same reconstruction, so
    /// the families built on the default cannot move.
    #[test]
    fn full_rank_is_byte_identical_to_no_rank() {
        let v = with_spectrum(200, &[4.0, 2.0, 1.0, 0.25], 8);
        let square = PcaRotate::default().fit(v.view(), None);
        let full = PcaRotate::truncated(4).fit(v.view(), None);
        assert_eq!(square, full);
        let mut y = v.clone();
        PcaRotate::truncated(4).apply(&full, &mut y, &[]);
        assert_eq!(
            PcaRotate::default().reconstruct(&square, &[], Some(y.view())),
            PcaRotate::truncated(4).reconstruct(&full, &[], Some(y.view()))
        );
    }

    #[test]
    #[should_panic(expected = "rank 5 exceeds the input dim 4")]
    fn rank_above_dim_is_refused_at_fit() {
        let v = with_spectrum(20, &[4.0, 2.0, 1.0, 0.25], 9);
        PcaRotate::truncated(5).fit(v.view(), None);
    }
}
