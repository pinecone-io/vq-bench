//! CENTER: subtracts the center from every vector
//! -
//! Model: mu, the mean over the fit vectors
//! Code for vector x: empty
//! Apply: x --> x - mu
//! Reconstruct: y --> y + mu
//! Score: s --> s + <q, mu>

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::{coding, math, Primitive};

pub struct Center;

/// The mean mu, read from the model bytes.
fn mean(model: &[u8]) -> Array1<f32> {
    coding::unpack_model(model)
}

impl Primitive for Center {
    fn describe() -> &'static str {
        "subtract the mean over the fit set from every vector"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        coding::pack_model(vectors.mean_axis(Axis(0)).unwrap())
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        let (n_v, d) = vectors.dim();
        *vectors -= &mean(model).broadcast((n_v, d)).unwrap();
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_recons.expect("Center is not terminal").to_owned();
        let (n_v, d) = out.dim();
        out += &mean(model).broadcast((n_v, d)).unwrap();
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = child_scores.expect("Center is not terminal").to_owned();
        math::offset_rows(&mut out, queries.dot(&mean(model)).view()); // add <q, mu> per query
        out
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{AsQuantizer, CastUint, MinMax, Pipeline, Quantizer};
    use ndarray::array;

    #[test]
    fn subtract_then_restore_and_score() {
        let v = array![[3., 5., 0.], [1., 3., 4.], [-1., 1., 2.], [1., 3., 2.]];
        let q = array![[1., 0., -1.], [0.5, 1., 0.]];
        let c = Center;
        let model = c.fit(v.view(), None);
        let mut x = v.clone();
        c.apply(&model, &mut x, &[]);
        // mean removed
        assert_close(
            &x.mean_axis(Axis(0)).unwrap().insert_axis(Axis(0)),
            &Array2::zeros((1, 3)),
            1e-4,
        );
        assert_close(&c.reconstruct(&model, &[], Some(x.view())), &v, 1e-4);
        // score on top of a lossless child recovers the exact dot.
        let child = q.dot(&x.t());
        assert_close(
            &c.score(&model, q.view(), &[], Some(child.view())),
            &q.dot(&v.t()),
            1e-3,
        );
    }

    #[test]
    fn composes_in_pipeline() {
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(
            Pipeline::new(
                4,
                vec![
                    Box::new(Center) as Box<dyn Primitive>,
                    Box::new(MinMax::default()),
                    Box::new(CastUint::new(8)),
                ],
            )
            .unwrap(),
        );
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
