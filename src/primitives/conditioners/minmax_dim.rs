//! MINMAXDIM: rescales each dimension [min, max] into [lo, hi]
//! -
//! Model: per-dimension (scale, offset) mapping [min_j, max_j] onto [lo, hi]
//! Code for vector x: empty
//! Apply: x_j --> scale_j * x_j + offset_j
//! Reconstruct: y_j --> (y_j - offset_j) / scale_j
//! Score: s --> s + <q, -offset/scale>
//!
use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

use crate::{coding, math, Primitive};

pub struct MinMaxDim {
    lo: f32,
    hi: f32,
}

impl MinMaxDim {
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }
}

impl Default for MinMaxDim {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

/// The per-dimension (scales, offsets) read from the model.
fn params(model: &[u8]) -> (Array1<f32>, Array1<f32>) {
    coding::unpack_model(model)
}

/// The inverse affine `(inv, bias)` from the model: inverts `x' = scale*x + offset`
/// per dimension as `x = inv*x' + bias`
fn inverse_affine(model: &[u8]) -> (Array1<f32>, Array1<f32>) {
    let (scales, offsets) = params(model);
    let inv = math::reciprocal(scales.view());
    let bias = Zip::from(&scales)
        .and(&offsets)
        .map_collect(|&scale, &offset| if scale != 0.0 { -offset / scale } else { offset });
    (inv, bias)
}

impl Primitive for MinMaxDim {
    fn describe() -> &'static str {
        "affine scale each dimension into the target range, calibrated over the fit set"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let min = vectors.map_axis(Axis(0), |c| c.iter().copied().fold(f32::INFINITY, f32::min));
        let max = vectors.map_axis(Axis(0), |c| {
            c.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        });
        // scale maps each dimension's [min, max] onto [lo, hi]
        let scales = (&max - &min).mapv(|span| {
            if span != 0.0 {
                (self.hi - self.lo) / span
            } else {
                0.0
            }
        });
        let offsets = Zip::from(&scales)
            .and(&min)
            .map_collect(|&scale, &min| self.lo - scale * min);
        coding::pack_model((scales, offsets))
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        let (scales, offsets) = params(model);
        math::affine_cols(vectors, scales.view(), offsets.view());
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        // q_j --> q_j / scale_j, so the downstream score is appropriately reweighted.
        let (scales, _) = params(model);
        math::scale_cols(queries, math::reciprocal(scales.view()).view());
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("MinMaxDim is not terminal").to_owned();
        let (inv, bias) = inverse_affine(model);
        math::affine_cols(&mut out, inv.view(), bias.view());
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // The child already carries inv*<q, x'> (the query was reweighted in apply_queries);
        // <q, x> = that + <q, bias>, a per-query row offset.
        let mut out = child_scores.expect("MinMaxDim is not terminal").to_owned();
        let (_, bias) = inverse_affine(model);
        math::offset_rows(&mut out, queries.dot(&bias).view());
        out
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores};
    use ndarray::array;

    #[test]
    fn per_dim_scale_then_restore_and_score() {
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let mm = MinMaxDim::default();
        let model = mm.fit(v.view(), None);

        // Every coordinate lands in [0, 1] after the per-dimension rescale.
        let mut x = v.clone();
        mm.apply(&model, &mut x, &[]);
        for &val in x.iter() {
            assert!((0.0..=1.0).contains(&val), "coord {val} out of [0,1]");
        }
        assert_close(&mm.reconstruct(&model, &[], Some(x.view())), &v, 1e-4);

        // score() sees the original query; the child rides the transformed query, exactly
        // as the pipeline feeds it. On a lossless child this recovers the exact dot.
        let mut qn = q.clone();
        mm.apply_queries(&model, &mut qn);
        let child = qn.dot(&x.t()); // <q_next, x'>
        assert_close(
            &mm.score(&model, q.view(), &[], Some(child.view())),
            &q.dot(&v.t()),
            1e-3,
        );
    }

    #[test]
    fn constant_dimension_maps_to_lo() {
        // Column 0 is constant across the fit set: span 0 -> scale 0, collapses to lo.
        let v = array![[1., 5.], [1., 7.], [1., 3.]];
        let mm = MinMaxDim::new(0.0, 1.0);
        let model = mm.fit(v.view(), None);
        let mut x = v.clone();
        mm.apply(&model, &mut x, &[]);
        assert!(x.column(0).iter().all(|&c| c == 0.0), "constant dim -> lo");
    }

    #[test]
    fn composes_in_pipeline() {
        // scalar quantization: per-dim rescale then a uniform cast.
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        assert_pipeline_scores(
            vec![
                Box::new(MinMaxDim::default()) as Box<dyn Primitive>,
                Box::new(crate::CastUint::new(8)),
            ],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
