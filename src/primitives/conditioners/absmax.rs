//! `adjust(absmax)`: scale each vector by its max-abs into `[-1, 1]`, carrying
//! `maxⱼ|xⱼ|` as side info.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{coding, math, Primitive};

/// Map `x ↦ x / maxⱼ|xⱼ|` so every coordinate lands in `[-1, 1]`; the per-vector
/// scale `maxⱼ|xⱼ|` is emitted as side info and folded back on reconstruct/score.
/// Queries are left untouched (asymmetric), so a child returning `⟨q, x/s⟩` becomes
/// `⟨q, x⟩` after the `×s` fold. A zero vector gets scale `0` and reconstructs to zero.
/// Symmetric counterpart of [`Normalize`](super::Normalize) (L∞ instead of L2).
pub struct AbsMax;

/// The per-vector scale carried in the codes.
fn scales(codes: &[&[u8]]) -> Array1<f32> {
    let [scale] = coding::unpack_f32_fields(codes);
    scale
}

impl Primitive for AbsMax {
    // fit uses the trait default (no model); the per-vector scale lives in the codes.

    fn encode(&self, _model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let s = vectors
            .mapv(f32::abs)
            .fold_axis(Axis(1), 0.0f32, |&a, &b| a.max(b));
        coding::pack_f32_fields([&s])
    }

    fn apply(&self, _model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        let inv = scales(codes).mapv(|s| if s > 0.0 { 1.0 / s } else { 0.0 });
        let zero = Array1::zeros(inv.len());
        math::affine_rows(vectors, &inv, &zero);
    }

    fn reconstruct(
        &self,
        _model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // x = s · x̂ : scale each row back up by its max-abs.
        let mut out = child_recons.expect("AbsMax is not terminal").to_owned();
        let s = scales(codes);
        let zero = Array1::zeros(s.len());
        math::affine_rows(&mut out, &s, &zero);
        out
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // ⟨q, x⟩ = s · ⟨q, x̂⟩ : scale each candidate column by its max-abs.
        let mut out = child_scores.expect("AbsMax is not terminal").to_owned();
        math::scale_cols(&mut out, &scales(codes));
        out
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(4) // one f32 scale per vector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{AsQuantizer, Pipeline, Quantizer};
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
        let child = q.dot(&xhat.t()); // ⟨q, x̂⟩
        assert_close(
            &am.score(&[], q.view(), &r, Some(child.view())),
            &q.dot(&v.t()),
            1e-3,
        );
    }

    #[test]
    fn composes_in_pipeline() {
        use crate::CastUint;
        // absmax maps to [-1,1]; MinMax then shifts into [0,1] for CastUint. Confirm the
        // conditioner stack round-trips on the score invariant against the pipeline's own
        // reconstruction.
        let v = array![[0., 1., 2., 3.], [4., -6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(Pipeline::new(vec![
            Box::new(AbsMax) as Box<dyn Primitive>,
            Box::new(crate::MinMax::default()),
            Box::new(CastUint::new(8)),
        ]));
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(
            &codec.score(&model, q.view(), &r),
            &q.dot(&codec.reconstruct(&model, &r).t()),
            1e-3,
        );
    }
}
