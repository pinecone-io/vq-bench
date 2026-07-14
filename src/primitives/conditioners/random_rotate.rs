//! `random_rotate`: a fixed, data-independent orthogonal rotation.

use ndarray::{Array2, ArrayView2};

use crate::{coding, math, Primitive};

/// Rotate every vector by a fixed `d×d` Haar-random orthogonal matrix `R` (the QR
/// factor of a Gaussian, all entropy from the seed). Orthogonal, so the inverse is
/// `Rᵀ`, the round-trip is exact, and dot products / norms are preserved. No
/// per-vector code. This is the dense `full` variant (`O(d²)` space/time); the
/// `hadamard`/`jl` variants will join this file later.
pub struct RandomRotate {
    seed: u64,
}

impl RandomRotate {
    /// A full dense rotation seeded by `seed`.
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    /// Read the `d×d` rotation back out of the model bytes.
    fn rotation(model: &[u8], d: usize) -> Array2<f32> {
        Array2::from_shape_vec((d, d), coding::read_f32s(model)).unwrap()
    }
}

impl Primitive for RandomRotate {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let d = vectors.ncols();
        let g = math::gaussian(&mut math::seed(self.seed), (d, d));
        let r = math::qr_q(g.view());
        coding::f32s_to_bytes(r)
    }

    // encode omitted: a rotation owns no per-vector bits (code_bytes = Some(0)).

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        let r = Self::rotation(model, vectors.ncols());
        *vectors = math::matmul(vectors.view(), r.view());
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        let r = Self::rotation(model, queries.ncols());
        *queries = math::matmul(queries.view(), r.view()); // same rotation preserves ⟨q, x⟩
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let child = child_recons.expect("RandomRotate is not terminal");
        let r = Self::rotation(model, child.ncols());
        math::matmul(child, r.t()) // Rᵀ = R⁻¹
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Queries were rotated in apply_queries and candidate codes encode rotated
        // vectors — the rotation contributes nothing further. Pass the child up.
        child_scores
            .expect("RandomRotate is not terminal")
            .to_owned()
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(0) // fixed-width: no per-vector bits
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{AsQuantizer, CastUint, MinMax, Pipeline, Quantizer};
    use ndarray::array;

    #[test]
    fn deterministic_in_seed() {
        let v = array![[1., 2., 3., 4.], [4., 3., 2., 1.]];
        let a = RandomRotate::new(7).fit(v.view(), None);
        let b = RandomRotate::new(7).fit(v.view(), None);
        let c = RandomRotate::new(8).fit(v.view(), None);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn orthogonal_round_trip_and_dot() {
        // R orthogonal ⇒ exact round-trip and dot products preserved.
        let v = array![[1., 2., 3., 4.], [-1., 0., 2., 1.], [3., 1., -2., 0.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., -0.5]];
        let rot = RandomRotate::new(1);
        let model = rot.fit(v.view(), None);

        // apply then reconstruct (identity child) recovers the input.
        let mut x = v.clone();
        rot.apply(&model, &mut x, &[]);
        let recon = rot.reconstruct(&model, &[], Some(x.view()));
        assert_close(&recon, &v, 1e-4);

        // rotating both sides preserves ⟨q, x⟩.
        let mut qr = q.clone();
        rot.apply_queries(&model, &mut qr);
        assert_close(&qr.dot(&x.t()), &q.dot(&v.t()), 1e-3);
    }

    #[test]
    fn composes_in_pipeline() {
        // rotate → minmax → cast: round-trip within lattice error, exact asymmetric score.
        let v = array![[0., 1., 2., 3.], [4., 6., 8., 10.], [-2., 1., 0., 5.]];
        let q = array![[1., 0., -1., 2.], [0.5, 1., 0., 0.]];
        let codec = AsQuantizer(Pipeline::new(vec![
            Box::new(RandomRotate::new(3)) as Box<dyn Primitive>,
            Box::new(MinMax::default()),
            Box::new(CastUint::new(8)),
        ]));
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(&codec.reconstruct(&model, &r), &v, 0.1);
        assert_close(
            &codec.score(&model, q.view(), &r),
            &q.dot(&codec.reconstruct(&model, &r).t()),
            1e-3,
        );
    }
}
