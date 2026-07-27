//! RESIZE: zero-pads or truncates every vector to a fixed width, keeping its norm
//! -
//! Model: empty
//! Code for vector x: empty (both widths are config)
//! Apply: x --> [x, 0] (widening) or gain * x[..out_dim] (narrowing)
//! Reconstruct: y --> y[..in_dim] or [y, 0] / gain
//! Score: s --> s  (queries resized the same way)
//!
//! gain = sqrt(in_dim/out_dim) when narrowing: truncating an isotropic vector keeps
//! out_dim/in_dim of its energy, so the rescale makes E||x'||^2 = ||x||^2 either way.

use ndarray::{s, Array2, ArrayView2};

use crate::Primitive;

pub struct Resize {
    in_dim: usize,
    out_dim: usize,
}

impl Resize {
    /// A fixed `in_dim -> out_dim` widening (zero-pad) or narrowing (truncate).
    pub fn new(in_dim: usize, out_dim: usize) -> Self {
        debug_assert!(in_dim > 0 && out_dim > 0);
        Self { in_dim, out_dim }
    }

    /// The energy rescale: `sqrt(in_dim/out_dim)` when narrowing, else 1.
    fn gain(&self) -> f32 {
        (self.in_dim as f32 / self.out_dim as f32).max(1.0).sqrt()
    }

    /// `x`'s leading columns copied into a zeroed `width`-wide batch and scaled by
    /// `factor`: a zero-pad when `width` is wider, a truncation when narrower.
    fn resize(x: ArrayView2<f32>, width: usize, factor: f32) -> Array2<f32> {
        let kept = x.ncols().min(width);
        let mut out = Array2::zeros((x.nrows(), width));
        out.slice_mut(s![.., ..kept]).assign(&x.slice(s![.., ..kept]));
        out *= factor;
        out
    }
}

impl Primitive for Resize {
    // fit omitted: both widths are config, so the model is empty.

    // encode omitted: a resize owns no per-vector bits.

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        *vectors = Self::resize(vectors.view(), self.out_dim, self.gain());
    }

    fn apply_queries(&self, _model: &[u8], queries: &mut Array2<f32>) {
        *queries = Self::resize(queries.view(), self.out_dim, self.gain());
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let child = child_recons.expect("Resize is not terminal");
        // Exact when widening; the min-norm preimage when narrowing.
        Self::resize(child, self.in_dim, 1.0 / self.gain())
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Queries are resized the same way, so the child's scores pass through.
        child_scores.expect("Resize is not terminal").to_owned()
    }

    fn in_dim(&self) -> Option<usize> {
        Some(self.in_dim)
    }

    fn out_dim(&self, _in_dim: usize) -> usize {
        self.out_dim
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores};
    use crate::{math, CastUint, MinMax};
    use ndarray::{array, Axis};

    fn norms(x: &Array2<f32>) -> Vec<f32> {
        x.rows().into_iter().map(|r| r.dot(&r).sqrt()).collect()
    }

    #[test]
    fn model_and_codes_are_empty() {
        let v = array![[1., 2., 3., 4.]];
        let rs = Resize::new(4, 8);
        assert!(rs.fit(v.view(), None).is_empty());
        assert_eq!(rs.code_bytes(4), Some(0));
        assert_eq!(rs.in_dim(), Some(4));
        assert_eq!(rs.out_dim(4), 8);
    }

    #[test]
    fn widening_round_trip_and_dot() {
        // Zero-padding loses nothing: exact round-trip, norms and dot products preserved.
        let v = array![[1., 2., 3., 4.], [-1., 0., 2., 1.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., -0.5]];
        let rs = Resize::new(4, 7);

        let mut x = v.clone();
        rs.apply(&[], &mut x, &[]);
        assert_eq!(x.ncols(), 7);
        assert_close(&rs.reconstruct(&[], &[], Some(x.view())), &v, 1e-6);
        assert_eq!(norms(&x), norms(&v));

        // resizing both sides preserves <q, x>.
        let mut resized_queries = q.clone();
        rs.apply_queries(&[], &mut resized_queries);
        assert_close(&resized_queries.dot(&x.t()), &q.dot(&v.t()), 1e-4);
    }

    #[test]
    fn narrowing_keeps_norm_and_is_a_projector() {
        // The sqrt(in/out) gain keeps norms right on average over isotropic rows, and
        // re-applying the lifted reconstruction is a no-op.
        let v = math::gaussian(&mut math::seed(3), (200, 128));
        let rs = Resize::new(128, 32);

        let mut x = v.clone();
        rs.apply(&[], &mut x, &[]);
        assert_eq!(x.ncols(), 32);
        let ratio: f32 = norms(&x).iter().zip(norms(&v)).map(|(a, b)| a / b).sum::<f32>() / 200.0;
        assert!((ratio - 1.0).abs() < 0.05, "mean norm ratio {ratio}");

        let lifted = rs.reconstruct(&[], &[], Some(x.view()));
        assert_eq!(lifted.ncols(), 128);
        let mut again = lifted.clone();
        rs.apply(&[], &mut again, &[]);
        assert_close(&again, &x, 1e-4);
        // The lift is the min-norm preimage, so it is never longer than the input.
        for (a, b) in norms(&lifted).iter().zip(norms(&v)) {
            assert!(*a <= b + 1e-3, "lift norm {a} exceeds {b}");
        }
    }

    #[test]
    fn narrowing_drops_the_trailing_columns() {
        let v = array![[1., 2., 3., 4.]];
        let rs = Resize::new(4, 2);
        let mut x = v.clone();
        rs.apply(&[], &mut x, &[]);
        let gain = 2f32.sqrt(); // sqrt(4/2)
        assert_close(&x, &array![[gain, 2. * gain]], 1e-6);
        assert_close(
            &rs.reconstruct(&[], &[], Some(x.view())),
            &array![[1., 2., 0., 0.]],
            1e-6,
        );
    }

    #[test]
    fn equal_widths_are_the_identity() {
        let v = math::gaussian(&mut math::seed(4), (5, 16));
        let rs = Resize::new(16, 16);
        let mut x = v.clone();
        rs.apply(&[], &mut x, &[]);
        assert_close(&x, &v, 1e-6);
        assert_close(&rs.reconstruct(&[], &[], Some(x.view())), &v, 1e-6);
    }

    #[test]
    fn composes_in_pipeline() {
        // widen -> minmax -> cast: round-trip within lattice error, exact asymmetric score.
        let v = math::gaussian(&mut math::seed(1), (6, 8));
        let q = math::gaussian(&mut math::seed(2), (3, 8));
        assert_pipeline_scores(
            vec![
                Box::new(Resize::new(8, 32)) as Box<dyn Primitive>,
                Box::new(MinMax::default()),
                Box::new(CastUint::new(8)),
            ],
            v.view(),
            q.view(),
            Some(0.2),
            1e-2,
        );
    }

    #[test]
    fn axis_sanity() {
        // Guard the row-vector convention: apply must map n x d to n x out_dim.
        let v = math::gaussian(&mut math::seed(9), (7, 12));
        let rs = Resize::new(12, 20);
        let mut x = v.clone();
        rs.apply(&[], &mut x, &[]);
        assert_eq!(x.len_of(Axis(0)), 7);
        assert_eq!(x.len_of(Axis(1)), 20);
    }
}
