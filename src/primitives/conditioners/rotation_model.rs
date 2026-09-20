//! The three uses a conditioner makes of a `d x d` rotation held in its model.
//!
//! `pca_rotate`, `optimize_pq` and `optimize_signs` all learn a rotation, store it whole,
//! and apply it identically; only the fit differs. Sharing these keeps the rotation's
//! semantics in one place. `random_rotate` is deliberately not a caller: it regenerates
//! its matrix from a seed per dim rather than storing one.

use ndarray::{Array2, ArrayView2};

use crate::{coding, math};

/// Read the `d x d` rotation back out of the model bytes.
pub(crate) fn matrix(model: &[u8]) -> Array2<f32> {
    coding::unpack_model(model)
}

/// Rotate a batch in place: `m --> m R`.
pub(crate) fn rotate(model: &[u8], m: &mut Array2<f32>) {
    *m = math::matmul(m.view(), matrix(model).view());
}

/// Undo the rotation on a child's reconstruction: `y --> y R^T` (`R^T = R^-1`).
pub(crate) fn unrotate(model: &[u8], child: ArrayView2<f32>) -> Array2<f32> {
    math::matmul(child, matrix(model).t())
}
