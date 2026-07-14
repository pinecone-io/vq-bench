//! Seeded randomness. ChaCha8 is the one RNG the crate uses.

use ndarray::Array2;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};

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
}
