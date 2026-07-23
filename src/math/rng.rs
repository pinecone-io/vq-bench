//! Seeded randomness. ChaCha8 is the one RNG the crate uses.

use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

use super::linalg::qr_q;

/// The crate's standard RNG.
pub type Rng = rand_chacha::ChaCha8Rng;

/// A deterministic RNG from a seed.
pub fn seed(seed: u64) -> Rng {
    Rng::seed_from_u64(seed)
}

/// A `(rows, cols)` matrix of standard-normal samples.
pub fn gaussian(rng: &mut Rng, (rows, cols): (usize, usize)) -> Array2<f32> {
    Array2::from_shape_fn((rows, cols), |_| StandardNormal.sample(rng))
}

/// A `d × d` Haar-random orthogonal matrix (the `Q` factor of a Gaussian's QR).
pub fn random_orthogonal(rng: &mut Rng, d: usize) -> Array2<f32> {
    qr_q(gaussian(rng, (d, d)).view())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_draw() {
        let a = gaussian(&mut seed(7), (4, 3));
        let b = gaussian(&mut seed(7), (4, 3));
        assert_eq!(a, b);
        let c = gaussian(&mut seed(8), (4, 3));
        assert_ne!(a, c);
    }

    #[test]
    fn random_orthogonal_is_orthogonal_and_seeded() {
        let q = random_orthogonal(&mut seed(7), 5);
        let qtq = super::super::matmul(q.t(), q.view());
        for i in 0..5 {
            for j in 0..5 {
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!((qtq[[i, j]] - expect).abs() < 1e-5, "QᵀQ[{i},{j}] = {}", qtq[[i, j]]);
            }
        }
        assert_eq!(q, random_orthogonal(&mut seed(7), 5));
    }
}
