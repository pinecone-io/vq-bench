//! KMEANS(LLOYD, k): learned codebook of k centroids (Lloyd); round to nearest
//! -
//! Model: input dim d, then the k x d centroids (data-dependent, so in the model)
//! Code for vector x: index of the nearest centroid (ceil(log2 k) bits)
//! Apply: x --> x - centroid(k)            (residual for the next stage)
//! Reconstruct: y --> centroid(k) + y
//! Score: s --> <q, centroid(k)> + s
//!
//! The terminal rounder behind PQ (one instance per segment).

use ndarray::{Array2, ArrayView2, Axis};

use crate::coding::CodeLayout;
use crate::{coding, math, Primitive};

/// Lloyd k-means rounder: `centroids` codewords trained in `fit`, stored in the model.
pub struct Kmeans {
    centroids: usize,
    seed: u64,
}

/// Lloyd iterations run during `fit`.
const ITERS: usize = 25;

impl Kmeans {
    /// A `centroids`-codeword Lloyd codebook (`2..=256`), seeded init.
    pub fn new(centroids: usize, seed: u64) -> Self {
        debug_assert!((2..=256).contains(&centroids));
        Self { centroids, seed }
    }

    /// Bits to index the codebook: `ceil(log2(centroids))`.
    fn index_bits(&self) -> u8 {
        (self.centroids as u32).next_power_of_two().trailing_zeros() as u8
    }

    /// The `k x d` centroid codebook, read back from the model.
    fn centroids(model: &[u8]) -> Array2<f32> {
        let (_dim, centroids): (usize, Array2<f32>) = coding::unpack_model(model);
        centroids
    }

    /// The code layout: one packed centroid index (`dims = 1`, dim-independent).
    fn layout(&self) -> CodeLayout {
        CodeLayout::new().bits(1, self.index_bits())
    }

    /// Look each vector's code up to its centroid (the reconstruction this stage owns).
    fn dequant(&self, model: &[u8], codes: &[&[u8]]) -> Array2<f32> {
        let centroids = Self::centroids(model);
        let (idx, []) = self.layout().unpack::<0>(codes); // (n x 1) centroid indices
        let mut out = Array2::zeros((codes.len(), centroids.ncols()));
        for (i, row) in idx.outer_iter().enumerate() {
            out.row_mut(i).assign(&centroids.row(row[0] as usize));
        }
        out
    }
}

impl Primitive for Kmeans {
    fn fit(&self, vectors: ArrayView2<f32>, _queries: Option<ArrayView2<f32>>) -> Vec<u8> {
        let centroids = math::lloyd_kmeans(vectors, self.centroids, ITERS, self.seed);
        coding::pack_model((vectors.ncols(), centroids)) // dim first (convention), then centroids
    }

    fn encode(&self, model: &[u8], vectors: ArrayView2<f32>) -> Vec<Vec<u8>> {
        let centroids = Self::centroids(model);
        let idx = math::nearest_centroid(vectors, centroids.view()); // one index per vector
        self.layout().pack(idx.insert_axis(Axis(1)).view(), &[]) // (n x 1)
    }

    fn apply(&self, model: &[u8], vectors: &mut Array2<f32>, codes: &[&[u8]]) {
        *vectors -= &self.dequant(model, codes); // residual
    }

    fn reconstruct(
        &self,
        model: &[u8],
        codes: &[&[u8]],
        child_recons: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        let mut out = self.dequant(model, codes); // (n x d), d recovered from the centroids
        if let Some(child) = child_recons {
            out += &child;
        }
        out
    }

    fn score(
        &self,
        model: &[u8],
        queries: ArrayView2<f32>,
        codes: &[&[u8]],
        child_scores: Option<ArrayView2<f32>>,
    ) -> Array2<f32> {
        // ADC: score the k distinct centroids once, then gather by each code's index --
        // O(n_q*k*d + n_q*n_c) instead of re-expanding every candidate (O(n_q*n_c*d)).
        let centroids = Self::centroids(model);
        let table = math::matmul(queries, centroids.t()); // (n_q x k): <q, centroid_j>
        let (idx, []) = self.layout().unpack::<0>(codes); // (n_c x 1) centroid indices
        let mut out = Array2::zeros((queries.nrows(), codes.len()));
        for (cand, row) in idx.outer_iter().enumerate() {
            out.column_mut(cand).assign(&table.column(row[0] as usize));
        }
        if let Some(child) = child_scores {
            out += &child;
        }
        out
    }

    fn code_bytes(&self, _in_dim: usize) -> Option<usize> {
        Some(self.layout().byte_len()) // one packed centroid index
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::{assert_close, refs};
    use ndarray::array;

    // Four well-separated points; k=4 gives each its own centroid (exact round trip).
    fn data() -> Array2<f32> {
        array![[0., 0., 1.], [10., 1., 0.], [-5., 8., 2.], [3., -4., 9.]]
    }

    #[test]
    fn deterministic_model() {
        let v = data();
        let km = Kmeans::new(4, 7);
        assert_eq!(km.fit(v.view(), None), km.fit(v.view(), None));
    }

    #[test]
    fn round_trip_and_score_exact_when_k_covers_points() {
        let v = data();
        let q = array![[1., 0., -1.], [0.5, 1., 0.]];
        let km = Kmeans::new(4, 7); // k = 4 = number of points
        let model = km.fit(v.view(), None);
        let codes = km.encode(&model, v.view());
        let r = refs(&codes);
        assert_close(&km.reconstruct(&model, &r, None), &v, 1e-4);
        assert_close(&km.score(&model, q.view(), &r, None), &q.dot(&v.t()), 1e-3);
    }

    #[test]
    fn score_matches_reconstruct_dot_with_many_candidates() {
        // n_c > k: the ADC table path must equal dotting q against each candidate's centroid.
        let v = math::gaussian(&mut math::seed(1), (50, 6));
        let q = math::gaussian(&mut math::seed(2), (7, 6));
        let km = Kmeans::new(8, 9); // k = 8 < 50 candidates
        let model = km.fit(v.view(), None);
        let codes = km.encode(&model, v.view());
        let r = refs(&codes);
        let via_adc = km.score(&model, q.view(), &r, None);
        let via_dot = q.dot(&km.reconstruct(&model, &r, None).t());
        assert_close(&via_adc, &via_dot, 1e-4);
    }

    #[test]
    fn code_is_one_byte_per_vector() {
        let v = data();
        let km = Kmeans::new(4, 7); // 4 centroids -> 2-bit index
        let codes = km.encode(&km.fit(v.view(), None), v.view());
        assert_eq!(codes[0].len(), km.code_bytes(3).unwrap());
        assert_eq!(codes[0].len(), 1);
    }
}
