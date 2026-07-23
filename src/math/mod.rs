//! Numerical operations supplied to primitives.
//!
//! The backend (faer for linear algebra) lives here. Primitives call these
//! functions and never depend on a backend directly.

mod batch;
mod linalg;
mod rng;

pub use batch::{
    affine_rows, offset_rows, outer, reciprocal, row_minmax, scale_cols, scale_rows,
};
pub use linalg::matmul;
pub use rng::{random_orthogonal, seed};
