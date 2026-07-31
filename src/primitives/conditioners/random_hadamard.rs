//! RANDOM_HADAMARD: near-orthogonal random rotation via the randomized Hadamard
//! transform (FhtKac): O(d log d) time, O(d) state, vs random_rotate's O(d^2).
//! -
//! Model: the ±1 sign vectors, one per round, of length padded_dim
//! Code for vector x: empty
//! Apply: x --> R x  (x zero-padded from dim up to a multiple of 64)
//! Reconstruct: y --> R^T y, cropped back to dim  (R orthonormal)
//! Score: s --> s  (queries are rotated the same way)

use ndarray::{s, Array2, ArrayView2};

use crate::{coding, math, Primitive};

/// Rounds the RaBitQ FhtKacRotator uses.
const DEFAULT_ROUNDS: usize = 4;
/// Padded width is a multiple of this.
const PAD_MULTIPLE: usize = 64;

pub struct RandomHadamard {
    dim: usize,
    seed: u64,
    rounds: usize,
}

impl RandomHadamard {
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed, rounds: DEFAULT_ROUNDS }
    }

    /// `dim` rounded up to a multiple of `PAD_MULTIPLE`: the transform's working width.
    fn padded_dim(dim: usize) -> usize {
        dim.div_ceil(PAD_MULTIPLE) * PAD_MULTIPLE
    }

    /// Largest power of two `<= dim`: the Hadamard block width.
    fn trunc_dim(dim: usize) -> usize {
        if dim.is_power_of_two() {
            dim
        } else {
            dim.next_power_of_two() >> 1
        }
    }

    /// Zero-pad `x` to `width` columns.
    fn pad_cols(x: ArrayView2<f32>, width: usize) -> Array2<f32> {
        let mut padded = Array2::zeros((x.nrows(), width));
        padded.slice_mut(s![.., ..x.ncols()]).assign(&x);
        padded
    }

    /// Round `i`'s Hadamard block: whole width when padded == n, else the front block
    /// on even rounds and the tail block on odd rounds.
    fn block_hadamard(x: &mut Array2<f32>, n: usize, round: usize) {
        let padded = x.ncols();
        let start = if padded == n || round.is_multiple_of(2) { 0 } else { padded - n };
        math::hadamard(x.slice_mut(s![.., start..start + n]));
    }

    /// R: zero-pad, then per round apply the sign flip, Hadamard block, and Kac butterfly.
    fn forward(&self, signs: &Array2<f32>, x: ArrayView2<f32>) -> Array2<f32> {
        let (padded, n) = (signs.ncols(), Self::trunc_dim(self.dim));
        let mut y = Self::pad_cols(x, padded);
        for i in 0..signs.nrows() {
            math::scale_cols(&mut y, signs.row(i));
            Self::block_hadamard(&mut y, n, i);
            if padded != n {
                math::kac_walk(&mut y);
            }
        }
        y
    }

    fn transform(&self, model: &[u8], m: &mut Array2<f32>) {
        let signs: Array2<f32> = coding::unpack_model(model);
        *m = self.forward(&signs, m.view());
    }
}

impl Primitive for RandomHadamard {
    fn describe() -> &'static str {
        "fast near-orthogonal random rotation via the randomized Hadamard transform"
    }

    fn fit(&self, _vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let signs = math::rademacher(&mut math::seed(self.seed), (self.rounds, Self::padded_dim(self.dim)));
        coding::pack_model(signs)
    }

    // encode omitted: the transform owns no per-vector bits.

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, _codes: &[&[u8]]) {
        self.transform(model, vectors);
    }

    fn apply_queries(&self, model: &[u8], queries: &mut Array2<f32>) {
        self.transform(model, queries);
    }

    fn reconstruct(
        &self,
        model: &[u8],
        _codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let signs: Array2<f32> = coding::unpack_model(model);
        let n = Self::trunc_dim(self.dim);
        let padded = signs.ncols();
        let mut y = child_recons.expect("RandomHadamard is not terminal").to_owned();
        // Replay the forward ops in reverse; each is its own inverse.
        for i in (0..signs.nrows()).rev() {
            if padded != n {
                math::kac_walk(&mut y);
            }
            Self::block_hadamard(&mut y, n, i);
            math::scale_cols(&mut y, signs.row(i));
        }
        y.slice(s![.., ..self.dim]).to_owned()
    }

    fn score(
        &self,
        _model: &[u8],
        _queries: ArrayView2<f32>,
        _codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // Queries are rotated the same way, so the child's scores pass through.
        child_scores.expect("RandomHadamard is not terminal").to_owned()
    }

    fn in_dim(&self) -> Option<usize> {
        Some(self.dim)
    }

    fn out_dim(&self, in_dim: usize) -> usize {
        Self::padded_dim(in_dim)
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(0) // no per-vector bits: the signs live in the model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, assert_pipeline_scores};
    use crate::{CastUint, MinMax};

    #[test]
    fn deterministic_in_seed() {
        let v = math::gaussian(&mut math::seed(0), (2, 8));
        let first = RandomHadamard::new(8, 7).fit(v.view(), None);
        let repeat = RandomHadamard::new(8, 7).fit(v.view(), None);
        let different = RandomHadamard::new(8, 9).fit(v.view(), None);
        assert_eq!(first, repeat);
        assert_ne!(first, different);
    }

    #[test]
    fn orthonormal_round_trip_and_dot() {
        // d = 128: padded == n, whole-vector FHT, no Kac.
        // d = 100: padded 128, n 64, exercises pad + front/tail + Kac.
        for d in [128usize, 100] {
            let v = math::gaussian(&mut math::seed(1), (5, d));
            let q = math::gaussian(&mut math::seed(2), (3, d));
            let rh = RandomHadamard::new(d, 7);
            let model = rh.fit(v.view(), None);

            let mut x = v.clone();
            rh.apply(&model, &mut x, &[]);
            assert_eq!(x.ncols(), RandomHadamard::padded_dim(d));
            let recon = rh.reconstruct(&model, &[], Some(x.view()));
            assert_close(&recon, &v, 1e-3);

            // transforming both sides preserves <q, x>.
            let mut rotated_queries = q.clone();
            rh.apply_queries(&model, &mut rotated_queries);
            assert_close(&rotated_queries.dot(&x.t()), &q.dot(&v.t()), 1e-2);
        }
    }

    #[test]
    fn composes_in_pipeline() {
        // hadamard -> minmax -> cast: round-trip within lattice error, exact asymmetric score.
        let v = math::gaussian(&mut math::seed(1), (6, 100));
        let q = math::gaussian(&mut math::seed(2), (3, 100));
        assert_pipeline_scores(
            vec![
                Box::new(RandomHadamard::new(100, 3)) as Box<dyn Primitive>,
                Box::new(MinMax::default()),
                Box::new(CastUint::new(8)),
            ],
            v.view(),
            q.view(),
            Some(0.2),
            1e-2,
        );
    }
}
