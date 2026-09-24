//! The three uses a conditioner makes of a `d x k` orthonormal-column matrix held in its
//! model: a rotation when `k = d`, the first `k` coordinates of one when `k < d`.
//!
//! `pca_rotate`, `optimize_pq` and `optimize_signs` all learn such a matrix, store it
//! whole, and apply it identically; only the fit differs. Sharing these keeps the
//! semantics in one place. `random_rotate` is deliberately not a caller: it regenerates
//! its matrix from a seed per dim rather than storing one.

use ndarray::{Array2, ArrayView2};

use crate::{coding, math};

/// Read the `d x k` matrix back out of the model bytes.
pub(crate) fn matrix(model: &[u8]) -> Array2<f32> {
    coding::unpack_model(model)
}

/// Rotate a batch in place: `m --> m R` (`n x d --> n x k`).
pub(crate) fn rotate(model: &[u8], m: &mut Array2<f32>) {
    *m = math::matmul(m.view(), matrix(model).view());
}

/// Map a child's reconstruction back: `y --> y R^T` (`n x k --> n x d`). The inverse
/// when `k = d`; the projection onto the retained subspace when `k < d`.
pub(crate) fn unrotate(model: &[u8], child: ArrayView2<f32>) -> Array2<f32> {
    math::matmul(child, matrix(model).t())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::testing::assert_close;
    use ndarray::s;

    /// `rotate` then `unrotate` through a `d x k` matrix is the projection `x R R^T`,
    /// and the identity at `k = d`.
    #[test]
    fn unrotate_of_rotate_is_the_projection() {
        let (d, k) = (6, 3);
        let full = math::random_orthogonal(&mut math::seed(1), d);
        let x = math::gaussian(&mut math::seed(2), (5, d));

        let square = coding::pack_model(full.clone());
        let mut y = x.clone();
        rotate(&square, &mut y);
        assert_eq!(y.dim(), (5, d));
        assert_close(&unrotate(&square, y.view()), &x, 1e-4);

        let thin = full.slice(s![.., ..k]).to_owned();
        let model = coding::pack_model(thin.clone());
        let mut y = x.clone();
        rotate(&model, &mut y);
        assert_eq!(y.dim(), (5, k));
        let projection = math::matmul(x.view(), math::matmul(thin.view(), thin.t()).view());
        assert_close(&unrotate(&model, y.view()), &projection, 1e-4);
    }
}
