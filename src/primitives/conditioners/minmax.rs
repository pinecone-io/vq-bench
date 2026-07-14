//! Affine conditioners that map `x ↦ scale·x + offset`.

use ndarray::{Array1, Array2, ArrayView2, Axis, Zip};

use crate::{coding, math, Primitive};

/// Rescale each vector's `[min, max]` into `[lo, hi]`; `(scale, offset)` per
/// vector are emitted as side info. No model. A constant vector gets `scale = 0`
/// (its value is unrecoverable and reconstructs to `lo`).
pub struct MinMax {
    lo: f32,
    hi: f32,
}

impl MinMax {
    /// Map into the given target range.
    pub fn new(lo: f32, hi: f32) -> Self {
        Self { lo, hi }
    }
}

impl Default for MinMax {
    /// The unit interval `[0, 1]`.
    fn default() -> Self {
        Self::new(0.0, 1.0)
    }
}

/// The per-vector `(scale, offset)` columns carried in the codes.
fn params(codes: &[&[u8]]) -> (Array1<f32>, Array1<f32>) {
    let [scale, offset] = coding::unpack_f32_fields(codes);
    (scale, offset)
}

impl Primitive for MinMax {
    // fit uses the trait default (no model); the per-vector (scale, offset) lives in the codes.

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let (mins, maxs) = math::row_minmax(vectors);
        let span = self.hi - self.lo;
        let scale = (&maxs - &mins).mapv(|range| if range > 0.0 { span / range } else { 0.0 });
        let offset = (&mins * &scale).mapv(|ms| self.lo - ms); // lo − min·scale (= lo when scale=0)
        coding::pack_f32_fields([&scale, &offset])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let (scale, offset) = params(codes);
        math::affine_rows(vectors, &scale, &offset);
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Invert: x = x'/scale − offset/scale. Constant vectors (scale=0) recover lo.
        let mut out = child_recons.expect("MinMax is not terminal").to_owned();
        let (scale, offset) = params(codes);
        let inv = scale.mapv(|s| if s != 0.0 { 1.0 / s } else { 0.0 });
        let bias = Zip::from(&scale)
            .and(&offset)
            .map_collect(|&s, &o| if s != 0.0 { -o / s } else { o }); // o = lo when degenerate
        math::affine_rows(&mut out, &inv, &bias);
        out
    }

    fn score(
        &self,
        _model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // ⟨q, x⟩ = inv·⟨q, x'⟩ − (offset/scale)·Σq, per candidate (constant → offset·Σq).
        let mut out = child_scores.expect("MinMax is not terminal").to_owned();
        let (scale, offset) = params(codes);
        let inv = scale.mapv(|s| if s != 0.0 { 1.0 / s } else { 0.0 });
        let off_inv = Zip::from(&scale)
            .and(&offset)
            .map_collect(|&s, &o| if s != 0.0 { o / s } else { -o });
        let sum_q = queries.sum_axis(Axis(1));
        math::scale_cols(&mut out, &inv);
        out -= &math::outer(&sum_q, &off_inv);
        out
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{AsQuantizer, Pipeline, Quantizer};
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
        let child = q.dot(&applied(&mm, &v, &r).t()); // ⟨q, x'⟩
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
        assert_eq!(coding::read_f32s(&codes[0]), vec![0.0, 0.0]); // scale 0, offset = lo
        let xp = applied(&mm, &v, &refs(&codes));
        assert_eq!(xp, array![[0., 0., 0.]]); // collapses to lo; value lost
    }

    #[test]
    fn composes_with_cast() {
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(Pipeline::new(vec![
            Box::new(MinMax::default()) as Box<dyn Primitive>,
            Box::new(crate::CastUint::new(4)),
        ]));
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        let recon = codec.reconstruct(&model, &r);
        // Bin-center reconstruction is within a half-bin (scaled by each row's range).
        assert_close(&recon, &v, 0.4);
        // The asymmetric score must equal the exact dot with the pipeline's own
        // reconstruction — the tight invariant the conditioner corrections enforce.
        assert_close(&codec.score(&model, q.view(), &r), &q.dot(&recon.t()), 1e-4);
    }
}
