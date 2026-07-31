//! MINMAX: rescales each vector's [min, max] into [lo, hi]
//! -
//! Model: empty
//! Code for vector x: (scale, offset) mapping [min, max] onto [lo, hi]
//! Apply: x --> scale * x + offset
//! Reconstruct: y --> (y - offset) / scale
//! Score: s --> s / scale - (offset / scale) * sum(q)

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

use crate::coding::CodeLayout;
use crate::{math, Primitive};

pub struct MinMax {
    lo: f32,
    hi: f32,
}

impl MinMax {
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }
}

impl Default for MinMax {
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

/// The code layout: two trailing scalars (scale, offset), no bit levels.
fn layout() -> CodeLayout {
    CodeLayout::new().scalars(2)
}

/// The per-vector (scales, offsets) carried in the codes.
fn params(codes: &[&[u8]]) -> (Array1<f32>, Array1<f32>) {
    let (_, [scales, offsets]) = layout().unpack::<2>(codes);
    (scales, offsets)
}

/// The inverse affine `(inv, bias)` from the codes: inverts `x' = scale*x + offset`
/// as `x = inv*x' + bias`; a flat vector (scale 0) recovers to lo (= offset).
fn inverse_affine(codes: &[&[u8]]) -> (Array1<f32>, Array1<f32>) {
    let (scales, offsets) = params(codes);
    let inv = math::reciprocal(scales.view());
    let bias = Zip::from(&scales)
        .and(&offsets)
        .map_collect(|&scale, &offset| if scale != 0.0 { -offset / scale } else { offset });
    (inv, bias)
}

impl Primitive for MinMax {
    fn describe() -> &'static str {
        "affine scale each vector into desired target range"
    }

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (min, max) = math::row_minmax(vectors);
        // scale maps [min, max] onto [lo, hi]; a flat vector (span 0) gets scale 0,
        // and then offset falls out to lo.
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
        layout().pack_scalars(&[scales.view(), offsets.view()])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let (scales, offsets) = params(codes);
        math::affine_rows(vectors, scales.view(), offsets.view());
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("MinMax is not terminal").to_owned();
        let (inv, bias) = inverse_affine(codes);
        math::affine_rows(&mut out, inv.view(), bias.view());
        out
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_scores.expect("MinMax is not terminal").to_owned();
        // <q, x> = inv*<q, x'> + bias*sum(q), the same inverse affine as reconstruct.
        let (inv, bias) = inverse_affine(codes);
        let sum_q = queries.sum_axis(Axis(1));
        math::scale_cols(&mut out, inv.view());
        out += &math::outer(sum_q.view(), bias.view());
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

    // Transform a copy and hand it back as the (identity) child reconstruction.
    fn applied(mm: &MinMax, v: &Array2<f32>, r: &[&[u8]]) -> Array2<f32> {
        let mut x = v.clone();
        mm.apply(&[], &mut x, r);
        x
    }

    #[test]
    fn round_trip_recovers_input() {
        let v = array![[1., 2., 4.], [-1., 0., 3.], [0., 5., 2.]];
        let mm = MinMax::default();
        let codes = mm.encode(&[], v.view());
        let r = refs(&codes);
        let xp = applied(&mm, &v, &r);
        for &x in xp.iter() {
            assert!(
                (0.0..=1.0).contains(&x),
                "transformed value {x} out of [0,1]"
            );
        }
        assert_close(&mm.reconstruct(&[], &r, Some(xp.view())), &v, 1e-4);
    }

    #[test]
    fn score_matches_dot() {
        let v = array![[1., 2., 4.], [-1., 0., 3.], [0., 5., 2.]];
        let q = array![[1., 0., -1.], [0.5, 0.5, 0.5]];
        let mm = MinMax::default();
        let codes = mm.encode(&[], v.view());
        let r = refs(&codes);
        let child = q.dot(&applied(&mm, &v, &r).t()); // <q, x'>
        assert_close(
            &mm.score(&[], q.view(), &r, Some(child.view())),
            &q.dot(&v.t()),
            1e-4,
        );
    }

    #[test]
    fn constant_vector_maps_to_lo() {
        let v = array![[2., 2., 2.]];
        let mm = MinMax::new(0.0, 1.0);
        let codes = mm.encode(&[], v.view());
        let (_, [scales, offsets]) = layout().unpack::<2>(&refs(&codes));
        assert_eq!((scales[0], offsets[0]), (0.0, 0.0)); // scale 0, offset = lo
        let xp = applied(&mm, &v, &refs(&codes));
        assert_eq!(xp, array![[0., 0., 0.]]); // collapses to lo; value lost
    }

    #[test]
    fn composes_with_cast() {
        // Bin-center reconstruction within a half-bin, exact asymmetric score -- the
        // tight invariant the conditioner corrections enforce.
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        assert_pipeline_scores(
            vec![
                Box::new(MinMax::default()) as Box<dyn Primitive>,
                Box::new(crate::CastUint::new(4)),
            ],
            v.view(),
            q.view(),
            Some(0.4),
            1e-4,
        );
    }
}
