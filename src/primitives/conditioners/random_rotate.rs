//! RANDOM_ROTATE: rotates every vector by a fixed dxd Haar-random orthogonal matrix R
//! -
//! Model: the seed used to generate the rotation matrix.
//! Code for vector x: empty
//! Apply: x --> Rx
//! Reconstruct: y --> R^T * y   (R^T = R^-1)
//! Score: s --> s  (queries also rotated)

use std::sync::OnceLock;

use ndarray::{Array2, ArrayView2};

use crate::{coding, math, Primitive};

pub struct RandomRotate {
    seed: u64,
    rotation: OnceLock<Array2<f32>>,
}

impl RandomRotate {
    /// A full dense rotation seeded by `seed`.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rotation: OnceLock::new(),
        }
    }

    /// The `d x d` rotation: regenerated from the model's seed on first use, then
    /// cached. A stage sees one (model, dim) for its lifetime, so the cache is stable.
    fn rotation(&self, model: &[u8], d: usize) -> &Array2<f32> {
        self.rotation.get_or_init(|| {
            let seed: usize = coding::unpack_model(model);
            math::random_orthogonal(&mut math::seed(seed as u64), d)
        })
    }

    /// Rotate a batch in place: `m --> m R` (dim taken from the batch).
    fn rotate(&self, model: &[u8], m: &mut Array2<f32>) {
        let rotation = self.rotation(model, m.ncols());
        *m = math::matmul(m.view(), rotation.view());
    }
}

impl Primitive for RandomRotate {
    fn describe() -> &'static str {
        "apply a random orthogonal transformation to all vectors"
    }

    fn fit(&self, _vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        // Store the seed, not the matrix: the seed is the honest, self-describing
        // description of the rotation.
        coding::pack_model(self.seed as usize)
    }

    // encode omitted: a rotation owns no per-vector bits.

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        self.rotate(model, vectors);
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        self.rotate(model, queries);
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let child = child_recons.expect("RandomRotate is not terminal");
        let rotation = self.rotation(model, child.ncols());
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
        child_scores
            .expect("RandomRotate is not terminal")
            .to_owned()
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores};
    use crate::{CastUint, MinMax};
    use ndarray::array;

    #[test]
    fn deterministic_in_seed() {
        let v = array![[1., 2., 3., 4.], [4., 3., 2., 1.]];
        let first = RandomRotate::new(7).fit(v.view(), None);
        let repeat = RandomRotate::new(7).fit(v.view(), None);
        let different = RandomRotate::new(8).fit(v.view(), None);
        assert_eq!(first, repeat);
        assert_ne!(first, different);
    }

    #[test]
    fn model_holds_seed_not_matrix() {
        // The model is just the seed -- the dxd matrix is regenerated on demand,
        // never stored, so the model stays tiny regardless of dim.
        let v = array![[0f32; 64]];
        let model = RandomRotate::new(1).fit(v.view(), None);
        assert!(model.len() < 16, "model is {} bytes, expected the seed", model.len());
    }

    #[test]
    fn orthogonal_round_trip_and_dot() {
        // R orthogonal => exact round-trip and dot products preserved.
        let v = array![[1., 2., 3., 4.], [-1., 0., 2., 1.], [3., 1., -2., 0.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., -0.5]];
        let rot = RandomRotate::new(1);
        let model = rot.fit(v.view(), None);

        // apply then reconstruct (identity child) recovers the input.
        let mut x = v.clone();
        rot.apply(&model, &mut x, &[]);
        let recon = rot.reconstruct(&model, &[], Some(x.view()));
        assert_close(&recon, &v, 1e-4);

        // rotating both sides preserves <q, x>.
        let mut rotated_queries = q.clone();
        rot.apply_queries(&model, &mut rotated_queries);
        assert_close(&rotated_queries.dot(&x.t()), &q.dot(&v.t()), 1e-3);
    }

    #[test]
    fn composes_in_pipeline() {
        // rotate -> minmax -> cast: round-trip within lattice error, exact asymmetric score.
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        assert_pipeline_scores(
            vec![
                Box::new(RandomRotate::new(3)) as Box<dyn Primitive>,
                Box::new(MinMax::default()),
                Box::new(CastUint::new(8)),
            ],
            v.view(),
            q.view(),
            Some(0.1),
            1e-3,
        );
    }
}
