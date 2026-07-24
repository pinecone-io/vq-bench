//! k-means clustering for learned-codebook rounders. Training is delegated to
//! linfa-clustering (random init, Lloyd iterations); assignment stays in-house.

use linfa::traits::Fit;
use linfa::DatasetBase;
use linfa_clustering::{KMeans, KMeansInit};
use ndarray::{Array1, Array2, ArrayView2};
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256Plus;

use super::linalg::matmul;

/// Index of the nearest centroid (squared L2) for each point. Uses
/// `‖p−c‖² = ‖p‖² − 2⟨p,c⟩ + ‖c‖²` and drops the per-point `‖p‖²` (constant in `c`),
/// so the argmin is over `‖c‖² − 2⟨p,c⟩`.
pub fn nearest_centroid(points: ArrayView2<f32>, centroids: ArrayView2<f32>) -> Array1<u32> {
    let dots = matmul(points, centroids.t()); // (n × k)
    let cnorm: Array1<f32> = centroids.rows().into_iter().map(|c| c.dot(&c)).collect();
    dots.outer_iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_val = f32::INFINITY;
            for (k, &d) in row.iter().enumerate() {
                let val = cnorm[k] - 2.0 * d;
                if val < best_val {
                    best_val = val;
                    best = k;
                }
            }
            best as u32
        })
        .collect()
}

/// `k`-centroid codebook (`k × d`) over `points`, trained by linfa-clustering with
/// random-point init and up to `iters` Lloyd rounds. Deterministic given `seed_val`.
/// Random init (over k-means++) keeps fitting large codebooks on million-vector bases
/// practical -- k-means++ seeding is sequential O(n*k) per call.
pub fn lloyd_kmeans(points: ArrayView2<f32>, k: usize, iters: usize, seed_val: u64) -> Array2<f32> {
    let rng = Xoshiro256Plus::seed_from_u64(seed_val); // determinism from the harness seed
    let data = DatasetBase::from(points.to_owned()); // unsupervised: records only
    let model = KMeans::params_with_rng(k, rng)
        .init_method(KMeansInit::Random)
        .max_n_iterations(iters as u64)
        .n_runs(1) // single run, matching the old one-shot behavior
        .fit(&data)
        .expect("k-means fit");
    model.centroids().to_owned()
}
