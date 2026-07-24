//! Numerical operations supplied to primitives.
//!
//! The backend (faer for linear algebra) lives here. Primitives call these
//! functions and never depend on a backend directly.

mod batch;
mod linalg;
mod rng;
mod transforms;

pub use batch::{
    affine_cols, affine_rows, offset_rows, outer, reciprocal, row_minmax, scale_cols, scale_rows,
};
pub use linalg::matmul;
pub use rng::{rademacher, random_orthogonal, seed};
pub use transforms::{hadamard, kac_walk};

// `gaussian` currently has only test callers (primitive round-trip tests); the
// non-test build reaches randomness through `rademacher`/`random_orthogonal`.
#[cfg(test)]
pub use rng::gaussian;
