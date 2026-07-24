//! OPTIMIZE_SIGNS: learns a dxd orthogonal rotation R minimizing the sign-
//! quantization error ||sign(x R) - x R||^2 (the ITQ rotation).
//! -
//! Fit: from a Haar-random R (seeded), alternate B = sign(X R) and
//!      R = procrustes(X^T B) for ITERS steps.
//! Model: R, the learned orthogonal rotation
//! Code for vector x: empty
//! Apply: x --> x R
//! Reconstruct: y --> R^T * y   (R^T = R^-1)
//! Score: s --> s  (queries also rotated)

use ndarray::{Array2, ArrayView2};

use crate::{coding, math, Primitive};

/// ITQ alternation steps (Gong et al. 2013 use 50).
const ITERS: usize = 50;

pub struct OptimizeSigns {
    seed: u64,
}

impl OptimizeSigns {
    /// A learned sign-quantization rotation; `seed` drives the Haar init.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Read the `d x d` rotation back out of the model bytes.
    fn rotation(model: &[u8]) -> Array2<f32> {
        coding::unpack_model(model)
    }

    /// Rotate a batch in place: `m --> m R`.
    fn rotate(model: &[u8], m: &mut Array2<f32>) {
        *m = math::matmul(m.view(), Self::rotation(model).view());
    }
}

impl Primitive for OptimizeSigns {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let d = vectors.ncols();
        let mut rotation = math::random_orthogonal(&mut math::seed(self.seed), d);
        for _ in 0..ITERS {
            let rotated = math::matmul(vectors, rotation.view());
            let signs = rotated.mapv(|z| if z >= 0.0 { 1.0 } else { -1.0 });
            // Procrustes: R = argmax_orthogonal tr(R^T * X^T B) minimizes ||B - X R||.
            let cross = math::matmul(vectors.t(), signs.view());
            rotation = math::orthogonal_procrustes(cross.view());
        }
        coding::pack_model(rotation)
    }

    // encode omitted: a rotation owns no per-vector bits.

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        Self::rotate(model, vectors);
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        Self::rotate(model, queries);
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let child = child_recons.expect("OptimizeSigns is not terminal");
        let rotation = Self::rotation(model);
        math::matmul(child, rotation.t())
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Queries are rotated the same way, so the child's scores pass through.
        child_scores.expect("OptimizeSigns is not terminal").to_owned()
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(0) // no per-vector bits: the rotation lives in the model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores};
    use crate::CastSign;
    use ndarray::Array2;

    fn sign_error(x: &Array2<f32>, r: &Array2<f32>) -> f32 {
        let z = math::matmul(x.view(), r.view());
        let b = z.mapv(|v| if v >= 0.0 { 1.0 } else { -1.0 });
        (&z - &b).mapv(|e| e * e).sum()
    }

    #[test]
    fn deterministic_in_seed() {
        let v = math::gaussian(&mut math::seed(0), (30, 16));
        let a = OptimizeSigns::new(7).fit(v.view(), None);
        let b = OptimizeSigns::new(7).fit(v.view(), None);
        let c = OptimizeSigns::new(8).fit(v.view(), None);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn orthogonal_round_trip_and_dot() {
        let v = math::gaussian(&mut math::seed(1), (5, 16));
        let q = math::gaussian(&mut math::seed(2), (3, 16));
        let os = OptimizeSigns::new(7);
        let model = os.fit(v.view(), None);
        let mut x = v.clone();
        os.apply(&model, &mut x, &[]);
        assert_close(&os.reconstruct(&model, &[], Some(x.view())), &v, 1e-3);
        let mut rq = q.clone();
        os.apply_queries(&model, &mut rq);
        assert_close(&rq.dot(&x.t()), &q.dot(&v.t()), 1e-2);
    }

    #[test]
    fn learns_better_than_random_init() {
        // The ITQ alternation is monotone: the learned rotation's sign-quant error
        // is <= that of its own Haar init.
        let v = math::gaussian(&mut math::seed(1), (200, 16));
        let model = OptimizeSigns::new(5).fit(v.view(), None);
        let learned = OptimizeSigns::rotation(&model);
        let init = math::random_orthogonal(&mut math::seed(5), 16);
        assert!(sign_error(&v, &learned) <= sign_error(&v, &init) + 1e-3);
    }

    #[test]
    fn composes_with_cast_sign() {
        // Learned rotation then asymmetric sign cast: exact score vs the pipeline's
        // own reconstruction (sign is lossy, so no reconstruction tolerance).
        let v = math::gaussian(&mut math::seed(1), (6, 16));
        let q = math::gaussian(&mut math::seed(2), (3, 16));
        assert_pipeline_scores(
            vec![
                Box::new(OptimizeSigns::new(3)) as Box<dyn Primitive>,
                Box::new(CastSign),
            ],
            v.view(),
            q.view(),
            None,
            1e-3,
        );
    }
}
