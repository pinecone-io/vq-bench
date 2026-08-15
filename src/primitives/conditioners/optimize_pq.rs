//! OPTIMIZE_PQ: learns a dxd orthogonal rotation R minimizing product-quantization
//! reconstruction error
//! -
//! Fit: alternate between PQ-encoding to centroids and updating rotation to match centroids
//! Model: R, the learned orthogonal rotation
//! Code for vector x: empty
//! Apply: x --> x R
//! Reconstruct: y --> R^T * y   (R^T = R^-1)
//! Score: s --> s  (queries also rotated)
//!
//! The alternation starts from the identity and is locally optimal, so where it lands
//! depends on what it is handed: Ge et al. 2013's best-performing variant runs it behind
//! the parametric rotation (`pca_rotate` -> `balance_parts`).

use ndarray::{s, Array2, ArrayView2, Axis};

use crate::{coding, math, Primitive, SegmentSplit};

/// Lloyd iterations per segment codebook, per alternation step.
const KMEANS_ITERS: usize = 10;

pub struct OptimizePq {
    centroids: usize,
    section_dim: usize,
    iters: usize,
    seed: u64,
}

impl OptimizePq {
    /// A learned PQ rotation over `section_dim`-column segments with `centroids`
    /// codewords each, refined by `iters` alternation steps; `seed` drives the internal
    /// codebook init.
    pub fn new(centroids: usize, section_dim: usize, iters: usize, seed: u64) -> Self {
        debug_assert!(section_dim > 0 && (2..=256).contains(&centroids));
        Self { centroids, section_dim, iters, seed }
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

impl Primitive for OptimizePq {
    fn describe() -> &'static str {
        "learn an orthogonal rotation minimizing product-quantization error"
    }

    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let d = vectors.ncols();
        let segments = SegmentSplit::new(d, self.section_dim).bounds();
        let mut rotation = Array2::<f32>::eye(d);
        for _ in 0..self.iters {
            let rotated = math::matmul(vectors, rotation.view());
            // PQ reconstruction of the rotated data, segment by segment.
            let mut recon = Array2::<f32>::zeros(rotated.raw_dim());
            for (seg, &(start, end)) in segments.iter().enumerate() {
                let segment = rotated.slice(s![.., start..end]);
                let centroids = math::lloyd_kmeans(
                    segment,
                    self.centroids,
                    KMEANS_ITERS,
                    self.seed.wrapping_add(seg as u64),
                );
                let assign = math::nearest_centroid(segment, centroids.view());
                let idx: Vec<usize> = assign.iter().map(|&a| a as usize).collect();
                recon
                    .slice_mut(s![.., start..end])
                    .assign(&centroids.select(Axis(0), &idx));
            }
            // Procrustes: R = argmax_orthogonal tr(R^T * X^T Xhat) minimizes ||X R - Xhat||.
            let cross = math::matmul(vectors.t(), recon.view());
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
        let child = child_recons.expect("OptimizePq is not terminal");
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
        child_scores.expect("OptimizePq is not terminal").to_owned()
    }

    fn code_bytes(&self, _model: &[u8], _in_dim: usize) -> Option<usize> {
        Some(0) // no per-vector bits: the rotation lives in the model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use crate::{AsQuantizer, Kmeans, Pipeline, Quantizer, Split};
    use ndarray::Array2;

    /// Alternation steps the `opq` family defaults to.
    const DEFAULT_ITERS: usize = 15;

    /// The default non-parametric stage: 16 centroids, `DEFAULT_ITERS` steps.
    fn optimize(section_dim: usize, seed: u64) -> OptimizePq {
        OptimizePq::new(16, section_dim, DEFAULT_ITERS, seed)
    }

    /// Low-rank (strongly correlated) data, where a decorrelating rotation helps PQ.
    fn correlated(n: usize, d: usize, seed: u64) -> Array2<f32> {
        let g = math::gaussian(&mut math::seed(seed), (n, d / 4));
        let mix = math::gaussian(&mut math::seed(seed ^ 0xabc), (d / 4, d));
        math::matmul(g.view(), mix.view())
    }

    /// PQ reconstruction error of `x` under rotation `r` (fresh per-segment codebooks).
    fn pq_error(x: &Array2<f32>, r: &Array2<f32>, centroids: usize, section_dim: usize) -> f32 {
        let rotated = math::matmul(x.view(), r.view());
        let mut recon = Array2::<f32>::zeros(rotated.raw_dim());
        let (mut start, mut seg) = (0usize, 0u64);
        while start < x.ncols() {
            let end = (start + section_dim).min(x.ncols());
            let s = rotated.slice(s![.., start..end]);
            let c = math::lloyd_kmeans(s, centroids, KMEANS_ITERS, 99 + seg);
            let a = math::nearest_centroid(s, c.view());
            let idx: Vec<usize> = a.iter().map(|&v| v as usize).collect();
            recon.slice_mut(s![.., start..end]).assign(&c.select(Axis(0), &idx));
            start = end;
            seg += 1;
        }
        (&rotated - &recon).mapv(|e| e * e).sum()
    }

    #[test]
    fn deterministic_in_seed() {
        let v = correlated(80, 16, 0);
        let a = optimize(4, 7).fit(v.view(), None);
        let b = optimize(4, 7).fit(v.view(), None);
        let c = optimize(4, 8).fit(v.view(), None);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn orthogonal_round_trip_and_dot() {
        let v = correlated(30, 16, 1);
        let q = math::gaussian(&mut math::seed(2), (3, 16));
        let op = optimize(4, 7);
        let model = op.fit(v.view(), None);
        let mut x = v.clone();
        op.apply(&model, &mut x, &[]);
        assert_close(&op.reconstruct(&model, &[], Some(x.view())), &v, 1e-2);
        let mut rq = q.clone();
        op.apply_queries(&model, &mut rq);
        assert_close(&rq.dot(&x.t()), &q.dot(&v.t()), 1e-2);
    }

    #[test]
    fn reduces_pq_error_vs_random() {
        // The learned rotation lowers PQ reconstruction error below a random one.
        let v = correlated(200, 16, 1);
        let model = optimize(4, 5).fit(v.view(), None);
        let learned = OptimizePq::rotation(&model);
        let random = math::random_orthogonal(&mut math::seed(123), 16);
        assert!(pq_error(&v, &learned, 16, 4) <= pq_error(&v, &random, 16, 4));
    }

    /// With no alternation steps there is nothing to learn: the rotation is the identity
    /// the fit starts from, so an upstream stage's rotation is what survives.
    #[test]
    fn zero_iters_is_the_identity() {
        let v = correlated(200, 16, 4);
        let model = OptimizePq::new(16, 4, 0, 5).fit(v.view(), None);
        assert_eq!(OptimizePq::rotation(&model), Array2::<f32>::eye(16));
    }

    /// More alternation steps do not raise PQ error on the data they were fitted to.
    #[test]
    fn more_iters_do_not_hurt() {
        let v = correlated(200, 16, 6);
        let error = |iters| {
            let model = OptimizePq::new(16, 4, iters, 5).fit(v.view(), None);
            pq_error(&v, &OptimizePq::rotation(&model), 16, 4)
        };
        assert!(error(DEFAULT_ITERS) <= error(1), "15 steps worse than 1");
    }

    #[test]
    fn composes_with_pq() {
        // OptimizePq -> split(segment) -> per-segment Kmeans: asymmetric score is exact
        // against the pipeline's own reconstruction.
        let v = correlated(60, 16, 2);
        let q = math::gaussian(&mut math::seed(3), (5, 16));
        let split = Split::from_factory(SegmentSplit::new(16, 4), |b, branch_dim| {
            Pipeline::new(
                branch_dim,
                vec![Box::new(Kmeans::new(16, 42 + b as u64)) as Box<dyn Primitive>],
            )
            .unwrap()
        });
        let codec = AsQuantizer(
            Pipeline::new(
                16,
                vec![Box::new(optimize(4, 1)) as Box<dyn Primitive>, Box::new(split)],
            )
            .unwrap(),
        );
        let model = codec.fit(v.view(), None);
        let codes = codec.encode(&model, v.view());
        let r = refs(&codes);
        let recon = codec.reconstruct(&model, &r);
        assert_close(&codec.score(&model, q.view(), &r), &q.dot(&recon.t()), 1e-3);
    }
}
